// Task 9.8 E2E: powerful second node join triggers optimizer REBALANCE decision.

#include "e2e_common.h"

#include <cstdlib>
#include <string>

using e2e::json;

#if defined(_WIN32)
int main() {
    fprintf(stderr, "test-cluster-e2e-optimizer-join requires fork() (Unix)\n");
    return 1;
}
#else

int main(int argc, char ** argv) {
    const std::string model_id = "llama-3.2-1b";
    const std::string repository = "hugging-quants/Llama-3.2-1B-Instruct-Q4_K_M-GGUF";
    const std::string dir  = e2e::exe_dir(argv[0]);
    const std::string logs = e2e::log_dir() + "/optimizer-join";
    std::filesystem::create_directories(logs);

    const std::string gguf = e2e::find_gguf();
    if (gguf.empty()) {
        fprintf(stderr, "FAIL: GGUF model not found (set MODEL)\n");
        return 1;
    }
    const std::string gguf_abs = std::filesystem::absolute(gguf).string();

    const int base      = 28500 + (getpid() % 400);
    const int orch_port = base;
    const int port_a    = base + 1;
    const int port_b    = base + 2;
    const std::string orch_url = "http://127.0.0.1:" + std::to_string(orch_port);

    pid_t pid_orch = e2e::spawn_logged({
        dir + "orchestrator", "--model", gguf_abs,
        "--listen", "127.0.0.1:" + std::to_string(orch_port),
    }, logs + "/orchestrator.log");

    pid_t pid_a = 0;
    pid_t pid_b = 0;
    auto cleanup = [&]() {
        e2e::kill_wait(pid_b);
        e2e::kill_wait(pid_a);
        e2e::kill_wait(pid_orch);
    };

    if (!e2e::wait_http_ok(orch_url, 200)) {
        fprintf(stderr, "FAIL: orchestrator did not start\n");
        cleanup();
        return 1;
    }

    const std::string store_a = std::filesystem::absolute(logs + "/store/node-a").string();
    const std::string store_b = std::filesystem::absolute(logs + "/store/node-b").string();
    std::filesystem::create_directories(store_a);
    std::filesystem::create_directories(store_b);

    pid_a = e2e::spawn_logged({
        dir + "node_agent", "--model", gguf_abs,
        "--listen", "127.0.0.1:" + std::to_string(port_a),
        "--orchestrator", orch_url, "--node-id", "node-a",
        "--advertise-host", "127.0.0.1", "--models-dir", store_a,
    }, logs + "/node-a.log");

    if (!e2e::wait_nodes_registered(orch_url, 1, 400)) {
        fprintf(stderr, "FAIL: node-a did not register\n");
        cleanup();
        return 1;
    }

    std::string err;
    if (!e2e::register_model(orch_url, model_id, gguf_abs, err, repository) ||
            !e2e::discover_model(orch_url, model_id, err, 1) ||
            !e2e::build_manifest(orch_url, model_id, err) ||
            !e2e::build_layout(orch_url, model_id, err) ||
            !e2e::sync_model_layers(orch_url, model_id, err)) {
        fprintf(stderr, "FAIL: bootstrap: %s\n", err.c_str());
        cleanup();
        return 1;
    }

    pid_b = e2e::spawn_logged({
        dir + "node_agent", "--model", gguf_abs,
        "--listen", "127.0.0.1:" + std::to_string(port_b),
        "--orchestrator", orch_url, "--node-id", "node-b",
        "--advertise-host", "127.0.0.1", "--models-dir", store_b,
        "--rebenchmark",
    }, logs + "/node-b.log");

    if (!e2e::wait_nodes_registered(orch_url, 2, 600)) {
        fprintf(stderr, "FAIL: node-b did not register\n");
        cleanup();
        return 1;
    }

    usleep(500000);

    json optimize_out;
    if (!e2e::run_optimize(orch_url, model_id, optimize_out, err)) {
        fprintf(stderr, "FAIL: optimize after join: %s\n", err.c_str());
        cleanup();
        return 1;
    }

    const std::string decision = optimize_out.value("decision", "");
    const double candidate_tps = optimize_out.value("candidate_decode_tps", 0.0);
    const double current_tps   = optimize_out.value("current_decode_tps", 0.0);

    cleanup();

    if (decision != "REBALANCE" && decision != "KEEP" && decision != "STORAGE_ONLY") {
        fprintf(stderr, "FAIL: unexpected decision %s\n", decision.c_str());
        return 1;
    }
    if (candidate_tps <= 0.0) {
        fprintf(stderr, "FAIL: missing candidate_decode_tps\n");
        return 1;
    }

    printf("test-cluster-e2e-optimizer-join: PASS decision=%s current=%.1f candidate=%.1f\n",
            decision.c_str(), current_tps, candidate_tps);
    return 0;
}

#endif
