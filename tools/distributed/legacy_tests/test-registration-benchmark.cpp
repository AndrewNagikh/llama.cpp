// Integration test: node auto-benchmark registration via orchestrator HTTP API

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

static bool wait_http_ok(const std::string & url, int retries = 300) {
    for (int i = 0; i < retries; ++i) {
        httplib::Client cli(url.c_str());
        cli.set_connection_timeout(1, 0);
        const auto res = cli.Get("/health");
        if (res && res->status == 200) {
            return true;
        }
        usleep(200000);
    }
    return false;
}

static bool wait_nodes_with_benchmark(const std::string & url, int min_nodes, int retries = 300) {
    for (int i = 0; i < retries; ++i) {
        httplib::Client cli(url.c_str());
        cli.set_connection_timeout(1, 0);
        const auto res = cli.Get("/nodes");
        if (res && res->status == 200) {
            try {
                const json j = json::parse(res->body);
                if (!j.contains("nodes") || j["nodes"].size() < (size_t) min_nodes) {
                    usleep(500000);
                    continue;
                }
                bool all_ok = true;
                for (const auto & node : j["nodes"]) {
                    const double score = node.value("score", 0.0);
                    const double decode = node.value("decode_tps", 0.0);
                    const double prefill = node.value("prefill_tps", 0.0);
                    if (score <= 0.0 || decode <= 0.0 || prefill <= 0.0) {
                        all_ok = false;
                        break;
                    }
                }
                if (all_ok) {
                    return true;
                }
            } catch (...) {}
        }
        usleep(500000);
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
    fprintf(stderr, "test-registration-benchmark requires fork() (Unix)\n");
    return 1;
#else
    const char * model_path = argv[1];
    const std::string dir = exe_dir(argv[0]);
    const int base = 24000 + (getpid() % 500);
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

    if (!wait_http_ok(orch_url) || !wait_nodes_with_benchmark(orch_url, 1)) {
        fprintf(stderr, "test-registration-benchmark: node not registered with benchmark data\n");
        kill(pid_a, SIGTERM); kill(pid_orch, SIGTERM);
        return 1;
    }

    httplib::Client cli(orch_url.c_str());
    const auto res = cli.Get("/nodes");
    if (!res || res->status != 200) {
        fprintf(stderr, "test-registration-benchmark: GET /nodes failed\n");
        kill(pid_a, SIGTERM); kill(pid_orch, SIGTERM);
        return 1;
    }

    const json nodes = json::parse(res->body)["nodes"];
    const auto & node = nodes[0];
    const double score = node.value("score", 0.0);
    const double decode = node.value("decode_tps", 0.0);
    const double prefill = node.value("prefill_tps", 0.0);

    printf("registered node-a score=%.1f decode=%.1f prefill=%.1f\n", score, decode, prefill);

    kill(pid_a, SIGTERM); kill(pid_orch, SIGTERM);
    waitpid(pid_a, nullptr, 0); waitpid(pid_orch, nullptr, 0);

    if (score <= 0.0 || decode <= 0.0 || prefill <= 0.0) {
        fprintf(stderr, "test-registration-benchmark: invalid benchmark metrics\n");
        return 1;
    }

    printf("test-registration-benchmark: OK\n");
    return 0;
#endif
}
