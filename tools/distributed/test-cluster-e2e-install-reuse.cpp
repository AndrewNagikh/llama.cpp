// Task 9.7 - scenario: repeated synchronization must be a no-op when ready.
//
// Syncs layers once, then runs sync again and verifies install plan stays empty.

#include "e2e_common.h"

#include <cstdlib>
#include <string>

using e2e::json;

#if defined(_WIN32)
int main() {
    fprintf(stderr, "test-cluster-e2e-install-reuse requires fork() (Unix)\n");
    return 1;
}
#else

int main(int argc, char ** argv) {
    const std::string model_id = "llama-3.2-1b";
    const std::string repository = "hugging-quants/Llama-3.2-1B-Instruct-Q4_K_M-GGUF";
    const std::string dir  = e2e::exe_dir(argv[0]);
    const std::string logs = e2e::log_dir() + "/install-reuse";
    std::filesystem::create_directories(logs);

    const std::string gguf = e2e::find_gguf();
    if (gguf.empty()) {
        fprintf(stderr, "FAIL: GGUF model not found (set MODEL)\n");
        return 1;
    }
    const std::string gguf_abs = std::filesystem::absolute(gguf).string();

    const int base      = 27000 + (getpid() % 400);
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
    e2e::wait_http_ok("http://127.0.0.1:" + std::to_string(port_a), 200);

    std::string err;
    if (!e2e::register_model(orch_url, model_id, gguf_abs, err, repository)) {
        fprintf(stderr, "FAIL: model registration: %s\n", err.c_str());
        cleanup();
        return 1;
    }
    if (!e2e::discover_model(orch_url, model_id, err, 1)) {
        fprintf(stderr, "FAIL: model discovery: %s\n", err.c_str());
        cleanup();
        return 1;
    }
    if (!e2e::build_manifest(orch_url, model_id, err)) {
        fprintf(stderr, "FAIL: manifest build: %s\n", err.c_str());
        cleanup();
        return 1;
    }
    if (!e2e::build_layout(orch_url, model_id, err)) {
        fprintf(stderr, "FAIL: layout build: %s\n", err.c_str());
        cleanup();
        return 1;
    }

    if (!e2e::sync_model_layers(orch_url, model_id, err)) {
        fprintf(stderr, "FAIL: first sync failed: %s\n", err.c_str());
        cleanup();
        return 1;
    }

    const std::string node_url = "http://127.0.0.1:" + std::to_string(port_a);
    auto count_layers = [&]() -> size_t {
        json out; int s = 0;
        if (e2e::http_get(node_url, "/installed-layers?model=" + model_id, out, s) && s == 200) {
            if (out.contains("layers") && out["layers"].is_array()) {
                return out["layers"].size();
            }
        }
        return 0;
    };

    const size_t layers_before = count_layers();

    if (!e2e::sync_model_layers(orch_url, model_id, err)) {
        fprintf(stderr, "FAIL: second sync failed: %s\n", err.c_str());
        cleanup();
        return 1;
    }

    std::string plan_err;
    const bool empty_plan = e2e::build_install_plan(orch_url, model_id, plan_err, 0);
    const size_t layers_after = count_layers();

    cleanup();

    if (empty_plan && layers_before > 0 && layers_before == layers_after) {
        printf("test-cluster-e2e-install-reuse: PASS (layers=%zu, plan empty)\n", layers_before);
        return 0;
    }
    fprintf(stderr, "test-cluster-e2e-install-reuse: FAIL (empty_plan=%d layers_before=%zu layers_after=%zu)\n",
            empty_plan, layers_before, layers_after);
    return 1;
}

#endif // _WIN32
