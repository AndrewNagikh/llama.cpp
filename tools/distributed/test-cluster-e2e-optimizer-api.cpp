// Task 9.8 E2E: optimizer REST API after model is ready.

#include "e2e_common.h"

#include <cstdlib>
#include <string>

using e2e::json;

#if defined(_WIN32)
int main() {
    fprintf(stderr, "test-cluster-e2e-optimizer-api requires fork() (Unix)\n");
    return 1;
}
#else

int main(int argc, char ** argv) {
    const std::string model_id = "llama-3.2-1b";
    const std::string repository = "hugging-quants/Llama-3.2-1B-Instruct-Q4_K_M-GGUF";
    const std::string dir  = e2e::exe_dir(argv[0]);
    const std::string logs = e2e::log_dir() + "/optimizer-api";
    std::filesystem::create_directories(logs);

    const std::string gguf = e2e::find_gguf();
    if (gguf.empty()) {
        fprintf(stderr, "FAIL: GGUF model not found (set MODEL)\n");
        return 1;
    }
    const std::string gguf_abs = std::filesystem::absolute(gguf).string();

    const int base      = 28000 + (getpid() % 400);
    const int orch_port = base;
    const int port_a    = base + 1;
    const std::string orch_url = "http://127.0.0.1:" + std::to_string(orch_port);
    const std::string store = std::filesystem::absolute(logs + "/store/node-a").string();
    std::filesystem::create_directories(store);

    pid_t pid_orch = e2e::spawn_logged({
        dir + "orchestrator", "--model", gguf_abs,
        "--listen", "127.0.0.1:" + std::to_string(orch_port),
    }, logs + "/orchestrator.log");

    pid_t pid_a = 0;
    auto cleanup = [&]() { e2e::kill_wait(pid_a); e2e::kill_wait(pid_orch); };

    if (!e2e::wait_http_ok(orch_url, 200)) {
        fprintf(stderr, "FAIL: orchestrator did not start\n");
        cleanup();
        return 1;
    }

    pid_a = e2e::spawn_logged({
        dir + "node_agent", "--model", gguf_abs,
        "--listen", "127.0.0.1:" + std::to_string(port_a),
        "--orchestrator", orch_url, "--node-id", "node-a",
        "--advertise-host", "127.0.0.1", "--models-dir", store,
    }, logs + "/node-a.log");

    if (!e2e::wait_nodes_registered(orch_url, 1, 400)) {
        fprintf(stderr, "FAIL: node did not register\n");
        cleanup();
        return 1;
    }

    std::string err;
    if (!e2e::register_model(orch_url, model_id, gguf_abs, err, repository)) {
        fprintf(stderr, "FAIL: register: %s\n", err.c_str());
        cleanup();
        return 1;
    }
    if (!e2e::discover_model(orch_url, model_id, err, 1)) {
        fprintf(stderr, "FAIL: discover: %s\n", err.c_str());
        cleanup();
        return 1;
    }
    if (!e2e::build_manifest(orch_url, model_id, err)) {
        fprintf(stderr, "FAIL: manifest: %s\n", err.c_str());
        cleanup();
        return 1;
    }
    if (!e2e::build_layout(orch_url, model_id, err)) {
        fprintf(stderr, "FAIL: layout: %s\n", err.c_str());
        cleanup();
        return 1;
    }

    json optimize_out;
    if (!e2e::run_optimize(orch_url, model_id, optimize_out, err)) {
        fprintf(stderr, "FAIL: optimize: %s\n", err.c_str());
        cleanup();
        return 1;
    }

    json stored;
    if (!e2e::get_optimization(orch_url, model_id, stored, err)) {
        fprintf(stderr, "FAIL: get optimization: %s\n", err.c_str());
        cleanup();
        return 1;
    }

    const std::string decision = optimize_out.value("decision", "");
    if (decision.empty()) {
        fprintf(stderr, "FAIL: empty decision\n");
        cleanup();
        return 1;
    }
    if (stored.value("decision", "") != decision) {
        fprintf(stderr, "FAIL: stored decision mismatch\n");
        cleanup();
        return 1;
    }

    cleanup();
    printf("test-cluster-e2e-optimizer-api: PASS decision=%s\n", decision.c_str());
    return 0;
}

#endif
