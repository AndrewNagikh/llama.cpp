// Integration test: orchestrator + 3 node_agent processes vs full model

#include "dist_common.h"
#include "split_gen3_common.h"
#include "split_gen_common.h"
#include "split_tcp_wire.h"

#include "httplib.h"
#include "nlohmann/json.hpp"
#include "llama.h"

#include "ggml-backend.h"

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
    cargs.reserve(args.size() + 1);
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

static bool http_post_json(const std::string & base, const std::string & path, const json & body, json & out, int & status) {
    httplib::Client cli(base.c_str());
    cli.set_connection_timeout(5, 0);
    cli.set_read_timeout(120, 0);
    const auto res = cli.Post(path.c_str(), body.dump(), "application/json");
    status = res ? res->status : 0;
    if (!res) {
        return false;
    }
    try {
        out = json::parse(res->body);
    } catch (...) {
        return false;
    }
    return true;
}

static bool gen3_send_recv(
        int ctrl_fd,
        split_gen_cmd cmd,
        int32_t n_tokens,
        int32_t pos_start,
        int32_t layer_end,
        const int32_t * tokens,
        split_gen3_a_resp & resp) {
    if (!split_gen_send_req(ctrl_fd, cmd, n_tokens, pos_start, layer_end, 0, tokens)) {
        return false;
    }
    return split_gen3_recv_a_resp(ctrl_fd, resp, nullptr);
}

static std::vector<llama_token> run_split3_baseline(const std::string & dir, const char * model_path, int max_new) {
    const split_gen3_layout layout = SPLIT_GEN3_LAYOUTS[0];
    const int base = 24000 + (getpid() % 500);
    const int bc_port   = base;
    const int ab_port   = base + 1;
    const int ctrl_port = base + 2;

    pid_t pid_a = 0, pid_b = 0, pid_c = 0;

    pid_c = fork();
    if (pid_c == 0) {
        const std::string bc = std::to_string(bc_port);
        const std::string ls = std::to_string(layout.layer_c_start);
        execl((dir + "split_gen3_c").c_str(), "split_gen3_c", model_path,
                "--bc-port", bc.c_str(), "--layer-start", ls.c_str(), (char *) nullptr);
        _exit(127);
    }

    usleep(500000);

    pid_b = fork();
    if (pid_b == 0) {
        const std::string ab = std::to_string(ab_port);
        const std::string bc = std::to_string(bc_port);
        const std::string ls = std::to_string(layout.layer_b_start);
        const std::string le = std::to_string(layout.layer_b_end);
        execl((dir + "split_gen3_b").c_str(), "split_gen3_b", model_path,
                "--ab-port", ab.c_str(), "--bc-port", bc.c_str(),
                "--layer-start", ls.c_str(), "--layer-end", le.c_str(), (char *) nullptr);
        _exit(127);
    }

    usleep(500000);

    pid_a = fork();
    if (pid_a == 0) {
        const std::string ctrl = std::to_string(ctrl_port);
        const std::string ab   = std::to_string(ab_port);
        const std::string le   = std::to_string(layout.layer_a_end);
        execl((dir + "split_gen3_a").c_str(), "split_gen3_a", model_path,
                "--ctrl-port", ctrl.c_str(), "--b-port", ab.c_str(),
                "--layer-end", le.c_str(), (char *) nullptr);
        _exit(127);
    }

    int ctrl_fd = -1;
    for (int retry = 0; retry < 300; ++retry) {
        ctrl_fd = split_tcp_connect("127.0.0.1", ctrl_port);
        if (ctrl_fd >= 0) {
            break;
        }
        usleep(100000);
    }

    std::vector<llama_token> out;
    if (ctrl_fd < 0) {
        kill(pid_a, SIGTERM);
        kill(pid_b, SIGTERM);
        kill(pid_c, SIGTERM);
        return out;
    }

    ggml_backend_load_all();
    llama_model * model = llama_model_load_from_file(model_path, llama_model_default_params());
    if (!model) {
        close(ctrl_fd);
        kill(pid_a, SIGTERM);
        kill(pid_b, SIGTERM);
        kill(pid_c, SIGTERM);
        return out;
    }

    const auto prompt = split_gen_tokenize(llama_model_get_vocab(model), SPLIT_GEN_PROMPT);
    llama_model_free(model);

    split_gen3_a_resp resp{};
    gen3_send_recv(ctrl_fd, SPLIT_GEN_CMD_RESET, 0, 0, layout.layer_a_end, nullptr, resp);

    std::vector<int32_t> ptoks(prompt.size());
    for (size_t i = 0; i < prompt.size(); ++i) {
        ptoks[i] = (int32_t) prompt[i];
    }

    if (!gen3_send_recv(ctrl_fd, SPLIT_GEN_CMD_PREFILL, (int32_t) prompt.size(), 0,
            layout.layer_a_end, ptoks.data(), resp)) {
        close(ctrl_fd);
        kill(pid_a, SIGTERM);
        kill(pid_b, SIGTERM);
        kill(pid_c, SIGTERM);
        return out;
    }

    out.push_back((llama_token) resp.token_id);
    llama_token cur = (llama_token) resp.token_id;

    for (int step = 1; step < max_new; ++step) {
        const int32_t tok = (int32_t) cur;
        const int32_t pos = (int32_t) prompt.size() + step - 1;
        if (!gen3_send_recv(ctrl_fd, SPLIT_GEN_CMD_DECODE, 1, pos, layout.layer_a_end, &tok, resp)) {
            break;
        }
        cur = (llama_token) resp.token_id;
        out.push_back(cur);
    }

    gen3_send_recv(ctrl_fd, SPLIT_GEN_CMD_SHUTDOWN, 0, 0, layout.layer_a_end, nullptr, resp);
    close(ctrl_fd);

    int st = 0;
    waitpid(pid_a, &st, 0);
    waitpid(pid_b, &st, 0);
    waitpid(pid_c, &st, 0);

    return out;
}

#endif

int main(int argc, char ** argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: %s MODEL.gguf [--test-node-failure]\n", argv[0]);
        return 1;
    }

#if defined(_WIN32)
    fprintf(stderr, "test-orchestrator-3node requires fork() (Unix)\n");
    return 1;
#else
    const char * model_path = argv[1];
    bool test_failure = false;
    for (int i = 2; i < argc; ++i) {
        if (strcmp(argv[i], "--test-node-failure") == 0) {
            test_failure = true;
        }
    }

    const std::string dir = exe_dir(argv[0]);
    const int base = 22000 + (getpid() % 500);

    const int orch_port = base;
    const int port_a    = base + 1;
    const int port_b    = base + 2;
    const int port_c    = base + 3;

    const std::string orch_url = "http://127.0.0.1:" + std::to_string(orch_port);

    pid_t pid_orch = spawn_bg({
        dir + "orchestrator",
        "--model", model_path,
        "--listen", "127.0.0.1:" + std::to_string(orch_port),
    });

    pid_t pid_a = spawn_bg({
        dir + "node_agent",
        "--model", model_path,
        "--listen", "127.0.0.1:" + std::to_string(port_a),
        "--orchestrator", orch_url,
        "--node-id", "node-a",
    });

    pid_t pid_b = spawn_bg({
        dir + "node_agent",
        "--model", model_path,
        "--listen", "127.0.0.1:" + std::to_string(port_b),
        "--orchestrator", orch_url,
        "--node-id", "node-b",
    });

    pid_t pid_c = spawn_bg({
        dir + "node_agent",
        "--model", model_path,
        "--listen", "127.0.0.1:" + std::to_string(port_c),
        "--orchestrator", orch_url,
        "--node-id", "node-c",
    });

    if (!wait_http_ok(orch_url) || !wait_http_ok("http://127.0.0.1:" + std::to_string(port_a)) ||
        !wait_http_ok("http://127.0.0.1:" + std::to_string(port_b)) ||
        !wait_http_ok("http://127.0.0.1:" + std::to_string(port_c))) {
        fprintf(stderr, "test-orchestrator-3node: services failed to start\n");
        kill(pid_a, SIGTERM);
        kill(pid_b, SIGTERM);
        kill(pid_c, SIGTERM);
        kill(pid_orch, SIGTERM);
        return 1;
    }

    json create_body = { { "model", "llama-3.2-1b" } };
    json create_out;
    int status = 0;

    if (!http_post_json(orch_url, "/session/create", create_body, create_out, status) || status != 200) {
        fprintf(stderr, "test-orchestrator-3node: session/create failed status=%d\n", status);
        kill(pid_a, SIGTERM);
        kill(pid_b, SIGTERM);
        kill(pid_c, SIGTERM);
        kill(pid_orch, SIGTERM);
        return 1;
    }

    const std::string session_id = create_out.value("session_id", "");
    printf("session_id=%s\n", session_id.c_str());

    if (test_failure) {
        httplib::Client nb(("http://127.0.0.1:" + std::to_string(port_b)).c_str());
        nb.set_read_timeout(5, 0);
        nb.Post("/shutdown");
        usleep(500000);
    }

    json gen_body = {
        { "session_id", session_id },
        { "prompt", SPLIT_GEN_PROMPT },
        { "max_tokens", DIST_MAX_NEW_TOKENS },
    };
    json gen_out;
    {
        httplib::Client cli(orch_url.c_str());
        cli.set_connection_timeout(5, 0);
        cli.set_read_timeout(test_failure ? 15 : 120, 0);
        const auto res = cli.Post("/session/generate", gen_body.dump(), "application/json");
        status = res ? res->status : 0;
        if (res) {
            try {
                gen_out = json::parse(res->body);
            } catch (...) {
                status = 0;
            }
        }
    }

    if (test_failure) {
        if (status == 200) {
            fprintf(stderr, "test-orchestrator-3node: expected failure when node B is down\n");
            kill(pid_a, SIGTERM);
            kill(pid_b, SIGTERM);
            kill(pid_c, SIGTERM);
            kill(pid_orch, SIGTERM);
            return 1;
        }
        printf("test-orchestrator-3node: graceful failure OK status=%d error=%s\n",
                status, gen_out.value("error", "").c_str());
        kill(pid_a, SIGTERM);
        kill(pid_b, SIGTERM);
        kill(pid_c, SIGTERM);
        kill(pid_orch, SIGTERM);
        waitpid(pid_a, nullptr, 0);
        waitpid(pid_b, nullptr, 0);
        waitpid(pid_c, nullptr, 0);
        waitpid(pid_orch, nullptr, 0);
        return 0;
    }

    if (status != 200) {
        fprintf(stderr, "test-orchestrator-3node: session/generate failed status=%d error=%s\n",
                status, gen_out.value("error", "").c_str());
        kill(pid_a, SIGTERM);
        kill(pid_b, SIGTERM);
        kill(pid_c, SIGTERM);
        kill(pid_orch, SIGTERM);
        return 1;
    }

    std::vector<llama_token> split_tokens;
    for (const auto & t : gen_out["tokens"]) {
        split_tokens.push_back((llama_token) t.get<int>());
    }

    const auto baseline_tokens = run_split3_baseline(dir, model_path, DIST_MAX_NEW_TOKENS);
    if (baseline_tokens.empty()) {
        fprintf(stderr, "test-orchestrator-3node: split baseline failed\n");
        kill(pid_a, SIGTERM);
        kill(pid_b, SIGTERM);
        kill(pid_c, SIGTERM);
        kill(pid_orch, SIGTERM);
        return 1;
    }

    const size_t n_cmp = std::min(baseline_tokens.size(), split_tokens.size());
    bool match = baseline_tokens.size() == split_tokens.size();
    for (size_t i = 0; i < n_cmp; ++i) {
        if (baseline_tokens[i] != split_tokens[i]) {
            match = false;
            fprintf(stderr, "MISMATCH step=%zu baseline=%d orchestrator=%d\n",
                    i, (int) baseline_tokens[i], (int) split_tokens[i]);
        }
    }

    printf("tokens=%zu match=%s text=%s\n", split_tokens.size(), match ? "TRUE" : "FALSE",
            gen_out.value("text", "").c_str());

    kill(pid_a, SIGTERM);
    kill(pid_b, SIGTERM);
    kill(pid_c, SIGTERM);
    kill(pid_orch, SIGTERM);
    waitpid(pid_a, nullptr, 0);
    waitpid(pid_b, nullptr, 0);
    waitpid(pid_c, nullptr, 0);
    waitpid(pid_orch, nullptr, 0);

    if (!match) {
        fprintf(stderr, "test-orchestrator-3node: FAILED\n");
        return 1;
    }

    printf("test-orchestrator-3node: OK (%zu tokens)\n", split_tokens.size());
    return 0;
#endif
}
