// Integration test: dynamic layer planner via orchestrator HTTP API

#include "layer_planner.h"
#include "split_gen_common.h"

#include "httplib.h"
#include "nlohmann/json.hpp"

#include <cstdio>
#include <cstring>
#include <filesystem>
#include <string>
#include <vector>

#if !defined(_WIN32)
#include <signal.h>
#include <sys/wait.h>
#include <unistd.h>
#endif

using json = nlohmann::json;

static std::string exe_dir(const char * argv0) {
    std::string p(argv0);
    const auto pos = p.find_last_of("/\\");
    if (pos != std::string::npos) {
        return p.substr(0, pos + 1);
    }
    return "./";
}

#if !defined(_WIN32)

static pid_t spawn_bg(const std::vector<std::string> & args) {
    const pid_t pid = fork();
    if (pid != 0) {
        return pid;
    }
    std::vector<char *> cargs;
    for (const auto & a : args) {
        cargs.push_back(const_cast<char *>(a.c_str()));
    }
    cargs.push_back(nullptr);
    execv(cargs[0], cargs.data());
    _exit(127);
}

static bool wait_http_ok(const std::string & url, int retries = 100) {
    for (int i = 0; i < retries; ++i) {
        httplib::Client cli(url.c_str());
        cli.set_connection_timeout(1, 0);
        const auto res = cli.Get("/health");
        if (res && res->status == 200) {
            return true;
        }
        usleep(100000);
    }
    return false;
}

static bool wait_nodes_registered(const std::string & url, int min_nodes, int retries = 300) {
    for (int i = 0; i < retries; ++i) {
        httplib::Client cli(url.c_str());
        cli.set_connection_timeout(1, 0);
        const auto res = cli.Get("/nodes");
        if (res && res->status == 200) {
            try {
                const json j = json::parse(res->body);
                if (j.contains("nodes") && j["nodes"].size() >= (size_t) min_nodes) {
                    return true;
                }
            } catch (...) {}
        }
        usleep(200000);
    }
    return false;
}

#endif

int main(int argc, char ** argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: %s MODEL.gguf\n", argv[0]);
        return 1;
    }

#if defined(_WIN32)
    fprintf(stderr, "test-orchestrator-dynamic-layout requires fork() (Unix)\n");
    return 1;
#else
    const char * model_path = argv[1];
    const std::string dir = exe_dir(argv[0]);
    const int base = 23000 + (getpid() % 500);
    const std::string orch_url = "http://127.0.0.1:" + std::to_string(base);

    pid_t pid_orch = spawn_bg({
        dir + "orchestrator",
        "--model", model_path,
        "--listen", "127.0.0.1:" + std::to_string(base),
    });

    pid_t pid_a = spawn_bg({
        dir + "node_agent", "--model", model_path,
        "--listen", "127.0.0.1:" + std::to_string(base + 1),
        "--orchestrator", orch_url,
        "--node-id", "node-a",
    });
    pid_t pid_b = spawn_bg({
        dir + "node_agent", "--model", model_path,
        "--listen", "127.0.0.1:" + std::to_string(base + 2),
        "--orchestrator", orch_url,
        "--node-id", "node-b",
    });
    pid_t pid_c = spawn_bg({
        dir + "node_agent", "--model", model_path,
        "--listen", "127.0.0.1:" + std::to_string(base + 3),
        "--orchestrator", orch_url,
        "--node-id", "node-c",
    });

    if (!wait_http_ok(orch_url) || !wait_nodes_registered(orch_url, 3)) {
        fprintf(stderr, "test-orchestrator-dynamic-layout: services/nodes not ready\n");
        kill(pid_a, SIGTERM); kill(pid_b, SIGTERM); kill(pid_c, SIGTERM); kill(pid_orch, SIGTERM);
        return 1;
    }

    httplib::Client cli(orch_url.c_str());
    cli.set_read_timeout(60, 0);

    // Task 9.1: model must be registered in the cluster registry before any
    // session can be created.
    const std::string model_id = "llama-3.2-1b";
    const std::string filename = std::filesystem::path(model_path).filename().string();
    json reg_body = {
        { "model_id", model_id },
        { "display_name", "Llama 3.2 1B Q4_K_M" },
        { "source", "huggingface" },
        { "repository", "hugging-quants/Llama-3.2-1B-Instruct-Q4_K_M-GGUF" },
        { "filename", filename },
        { "revision", "main" }
    };

    const auto reg_res = cli.Post("/models/register", reg_body.dump(), "application/json");
    if (!reg_res || reg_res->status != 200) {
        fprintf(stderr, "test-orchestrator-dynamic-layout: /models/register failed status=%d\n",
                reg_res ? reg_res->status : 0);
        kill(pid_a, SIGTERM); kill(pid_b, SIGTERM); kill(pid_c, SIGTERM); kill(pid_orch, SIGTERM);
        return 1;
    }
    json reg_out = json::parse(reg_res->body);
    if (reg_out.value("status", "") != "DISCOVERED") {
        fprintf(stderr, "test-orchestrator-dynamic-layout: unexpected status %s\n",
                reg_out.value("status", "").c_str());
        kill(pid_a, SIGTERM); kill(pid_b, SIGTERM); kill(pid_c, SIGTERM); kill(pid_orch, SIGTERM);
        return 1;
    }

    const auto get_res = cli.Get("/models/" + model_id);
    if (!get_res || get_res->status != 200) {
        fprintf(stderr, "test-orchestrator-dynamic-layout: GET /models/{id} failed\n");
        kill(pid_a, SIGTERM); kill(pid_b, SIGTERM); kill(pid_c, SIGTERM); kill(pid_orch, SIGTERM);
        return 1;
    }
    json get_out = json::parse(get_res->body);
    if (get_out.value("status", "") != "DISCOVERED") {
        fprintf(stderr, "test-orchestrator-dynamic-layout: model status is not DISCOVERED\n");
        kill(pid_a, SIGTERM); kill(pid_b, SIGTERM); kill(pid_c, SIGTERM); kill(pid_orch, SIGTERM);
        return 1;
    }

    // Task 9.2: run remote discovery before creating a session.
    const auto disc_res = cli.Post("/models/" + model_id + "/discover", "", "application/json");
    if (!disc_res || disc_res->status != 200) {
        fprintf(stderr, "test-orchestrator-dynamic-layout: /models/{id}/discover failed status=%d\n",
                disc_res ? disc_res->status : 0);
        kill(pid_a, SIGTERM); kill(pid_b, SIGTERM); kill(pid_c, SIGTERM); kill(pid_orch, SIGTERM);
        return 1;
    }
    json disc_out = json::parse(disc_res->body);
    if (disc_out.value("status", "") != "ok" || disc_out.value("files", 0) == 0) {
        fprintf(stderr, "test-orchestrator-dynamic-layout: discovery failed %s\n",
                disc_out.dump().c_str());
        kill(pid_a, SIGTERM); kill(pid_b, SIGTERM); kill(pid_c, SIGTERM); kill(pid_orch, SIGTERM);
        return 1;
    }

    const auto get2_res = cli.Get("/models/" + model_id);
    if (!get2_res || get2_res->status != 200) {
        fprintf(stderr, "test-orchestrator-dynamic-layout: GET /models/{id} after discovery failed\n");
        kill(pid_a, SIGTERM); kill(pid_b, SIGTERM); kill(pid_c, SIGTERM); kill(pid_orch, SIGTERM);
        return 1;
    }
    json get2_out = json::parse(get2_res->body);
    if (get2_out.value("status", "") != "MANIFEST_PENDING") {
        fprintf(stderr, "test-orchestrator-dynamic-layout: model status is not MANIFEST_PENDING\n");
        kill(pid_a, SIGTERM); kill(pid_b, SIGTERM); kill(pid_c, SIGTERM); kill(pid_orch, SIGTERM);
        return 1;
    }
    if (!get2_out.contains("files") || get2_out["files"].empty()) {
        fprintf(stderr, "test-orchestrator-dynamic-layout: registry has no files after discovery\n");
        kill(pid_a, SIGTERM); kill(pid_b, SIGTERM); kill(pid_c, SIGTERM); kill(pid_orch, SIGTERM);
        return 1;
    }

    const auto create = cli.Post("/session/create",
            std::string("{\"model\":\"") + model_id + "\"}", "application/json");
    if (!create || create->status != 200) {
        fprintf(stderr, "session/create failed status=%d\n", create ? create->status : 0);
        kill(pid_a, SIGTERM); kill(pid_b, SIGTERM); kill(pid_c, SIGTERM); kill(pid_orch, SIGTERM);
        return 1;
    }

    json create_out = json::parse(create->body);
    const auto layout = create_out["layout"];

    // With the Task 8 memory-aware planner a single node may be able to hold
    // the whole model.  Validate only that the layout is contiguous and covers
    // all layers, then trust /session/generate to exercise the pipeline.
    bool layout_ok = true;
    if (layout.empty() || layout.front()["start"].get<int>() != 0) {
        layout_ok = false;
    }
    int covered = 0;
    int cursor = 0;
    for (size_t i = 0; i < layout.size() && layout_ok; ++i) {
        const int start = layout[i].value("start", -1);
        const int end   = layout[i].value("end", -1);
        if (start != cursor || end < start || start < 0 || end > 16) {
            layout_ok = false;
            fprintf(stderr, "layout[%zu]: invalid range [%d,%d), cursor=%d\n",
                    i, start, end, cursor);
        }
        cursor = end;
        covered += end - start;
    }
    if (covered != 16 || cursor != 16) {
        layout_ok = false;
        fprintf(stderr, "test-orchestrator-dynamic-layout: layout coverage wrong covered=%d/16\n",
                covered);
    }

    if (!layout_ok) {
        fprintf(stderr, "test-orchestrator-dynamic-layout: layout invalid\n");
        kill(pid_a, SIGTERM); kill(pid_b, SIGTERM); kill(pid_c, SIGTERM); kill(pid_orch, SIGTERM);
        return 1;
    }

    const std::string session_id = create_out.value("session_id", "");
    json gen_body = {
        { "session_id", session_id },
        { "prompt", SPLIT_GEN_PROMPT },
        { "max_tokens", 32 },
    };

    const auto gen = cli.Post("/session/generate", gen_body.dump(), "application/json");
    if (!gen || gen->status != 200) {
        fprintf(stderr, "session/generate failed status=%d body=%s\n",
                gen ? gen->status : 0, gen ? gen->body.c_str() : "");
        kill(pid_a, SIGTERM); kill(pid_b, SIGTERM); kill(pid_c, SIGTERM); kill(pid_orch, SIGTERM);
        return 1;
    }

    json gen_out = json::parse(gen->body);
    const int count = gen_out.value("count", 0);
    printf("dynamic layout OK, generated %d tokens: %s\n",
            count, gen_out.value("text", "").c_str());

    kill(pid_a, SIGTERM); kill(pid_b, SIGTERM); kill(pid_c, SIGTERM); kill(pid_orch, SIGTERM);
    waitpid(pid_a, nullptr, 0); waitpid(pid_b, nullptr, 0);
    waitpid(pid_c, nullptr, 0); waitpid(pid_orch, nullptr, 0);

    if (count != 32) {
        fprintf(stderr, "test-orchestrator-dynamic-layout: expected 32 tokens, got %d\n", count);
        return 1;
    }

    printf("test-orchestrator-dynamic-layout: OK\n");
    return 0;
#endif
}
