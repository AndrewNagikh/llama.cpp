// Integration test: dynamic layer planner via orchestrator HTTP API

#include "layer_planner.h"
#include "split_gen_common.h"

#include "httplib.h"
#include "nlohmann/json.hpp"

#include <cstdio>
#include <cstring>
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

static bool wait_nodes_registered(const std::string & url, int min_nodes, int retries = 150) {
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
        "--node-id", "node-a", "--score", "100",
    });
    pid_t pid_b = spawn_bg({
        dir + "node_agent", "--model", model_path,
        "--listen", "127.0.0.1:" + std::to_string(base + 2),
        "--orchestrator", orch_url,
        "--node-id", "node-b", "--score", "50",
    });
    pid_t pid_c = spawn_bg({
        dir + "node_agent", "--model", model_path,
        "--listen", "127.0.0.1:" + std::to_string(base + 3),
        "--orchestrator", orch_url,
        "--node-id", "node-c", "--score", "25",
    });

    if (!wait_http_ok(orch_url) || !wait_nodes_registered(orch_url, 3)) {
        fprintf(stderr, "test-orchestrator-dynamic-layout: services/nodes not ready\n");
        kill(pid_a, SIGTERM); kill(pid_b, SIGTERM); kill(pid_c, SIGTERM); kill(pid_orch, SIGTERM);
        return 1;
    }

    httplib::Client cli(orch_url.c_str());
    cli.set_read_timeout(60, 0);

    const auto create = cli.Post("/session/create",
            R"({"model":"llama-3.2-1b"})", "application/json");
    if (!create || create->status != 200) {
        fprintf(stderr, "session/create failed status=%d\n", create ? create->status : 0);
        kill(pid_a, SIGTERM); kill(pid_b, SIGTERM); kill(pid_c, SIGTERM); kill(pid_orch, SIGTERM);
        return 1;
    }

    json create_out = json::parse(create->body);
    const auto layout = create_out["layout"];

    const auto expected = dist_plan_layers(16, {
        { "node-a", 100.0 }, { "node-b", 50.0 }, { "node-c", 25.0 },
    });

    bool layout_ok = layout.size() == expected.size();
    for (size_t i = 0; i < expected.size() && layout_ok; ++i) {
        const int start = layout[i].value("start", -1);
        const int end   = layout[i].value("end", -1);
        if (start != expected[i].layer_start || end != expected[i].layer_end) {
            layout_ok = false;
            fprintf(stderr, "layout[%zu]: got [%d,%d) expected [%d,%d)\n",
                    i, start, end, expected[i].layer_start, expected[i].layer_end);
        }
    }

    if (!layout_ok) {
        fprintf(stderr, "test-orchestrator-dynamic-layout: layout mismatch\n");
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
