// Task 7.5 - scenario: orchestrator state survives a restart.
//
// Starts orchestrator + 3 nodes, installs a model, restarts the orchestrator,
// and verifies the catalog persists, nodes reconnect (heartbeat), and the model
// is reported ready again via GET /models.

#include "e2e_common.h"

#include <string>

using e2e::json;

#if defined(_WIN32)
int main() {
    fprintf(stderr, "test-cluster-e2e-orchestrator-restart requires fork() (Unix)\n");
    return 1;
}
#else

int main(int argc, char ** argv) {
    const std::string model_id = "llama-3.2-1b";
    const std::string repository = "hugging-quants/Llama-3.2-1B-Instruct-Q4_K_M-GGUF";
    const std::string dir  = e2e::exe_dir(argv[0]);
    const std::string logs = e2e::log_dir() + "/orch-restart";
    std::filesystem::create_directories(logs);

    const std::string gguf = e2e::find_gguf();
    if (gguf.empty()) {
        fprintf(stderr, "FAIL: GGUF model not found (set MODEL)\n");
        return 1;
    }
    const std::string gguf_abs = std::filesystem::absolute(gguf).string();

    const int base      = 29000 + (getpid() % 400);
    const int orch_port = base;
    const int port_a    = base + 1;
    const int port_b    = base + 2;
    const int port_c    = base + 3;
    const std::string orch_url = "http://127.0.0.1:" + std::to_string(orch_port);
    const std::string store = std::filesystem::absolute(logs + "/store").string();

    pid_t pid_orch = 0, pid_a = 0, pid_b = 0, pid_c = 0;
    auto cleanup = [&]() {
        e2e::kill_wait(pid_a); e2e::kill_wait(pid_b);
        e2e::kill_wait(pid_c); e2e::kill_wait(pid_orch);
    };

    auto start_orch = [&]() {
        return e2e::spawn_logged({
            dir + "orchestrator", "--model", gguf_abs,
            "--listen", "127.0.0.1:" + std::to_string(orch_port),
        }, logs + "/orchestrator.log");
    };

    pid_orch = start_orch();
    if (!e2e::wait_http_ok(orch_url, 200)) {
        fprintf(stderr, "FAIL: orchestrator did not start\n");
        cleanup();
        return 1;
    }

    auto spawn_node = [&](const std::string & id, int port) {
        const std::string s = store + "/" + id;
        std::filesystem::create_directories(s);
        return e2e::spawn_logged({
            dir + "node_agent", "--model", gguf_abs,
            "--listen", "127.0.0.1:" + std::to_string(port),
            "--orchestrator", orch_url, "--node-id", id,
            "--advertise-host", "127.0.0.1", "--models-dir", s,
        }, logs + "/" + id + ".log");
    };

    pid_a = spawn_node("node-a", port_a);
    pid_b = spawn_node("node-b", port_b);
    pid_c = spawn_node("node-c", port_c);

    if (!e2e::wait_nodes_registered(orch_url, 3, 400)) {
        fprintf(stderr, "FAIL: 3 nodes did not register\n");
        cleanup();
        return 1;
    }
    for (int p : { port_a, port_b, port_c }) {
        e2e::wait_http_ok("http://127.0.0.1:" + std::to_string(p), 200);
    }

    std::string err;
    if (!e2e::register_model(orch_url, model_id, gguf_abs, err, repository)) {
        fprintf(stderr, "FAIL: model registration: %s\n", err.c_str());
        cleanup();
        return 1;
    }
    std::string disc_err;
    if (!e2e::discover_model(orch_url, model_id, disc_err, 1)) {
        fprintf(stderr, "FAIL: model discovery: %s\n", disc_err.c_str());
        cleanup();
        return 1;
    }

    if (!e2e::install_and_wait_ready(orch_url, model_id, 3, err)) {
        fprintf(stderr, "FAIL: install did not complete: %s\n", err.c_str());
        cleanup();
        return 1;
    }

    // Capture catalog before restart.
    json cat_before; int cs = 0;
    const bool cat_ok_before = e2e::http_get(orch_url, "/catalog", cat_before, cs) && cs == 200 &&
                               cat_before.is_array() && !cat_before.empty();

    // Restart orchestrator.
    e2e::kill_wait(pid_orch);
    pid_orch = start_orch();
    if (!e2e::wait_http_ok(orch_url, 200)) {
        fprintf(stderr, "FAIL: orchestrator did not restart\n");
        cleanup();
        return 1;
    }

    const bool nodes_back = e2e::wait_nodes_registered(orch_url, 3, 400);
    const bool model_back = nodes_back && e2e::models_ready(orch_url, model_id, 3);

    json cat_after; int cs2 = 0;
    const bool cat_ok_after = e2e::http_get(orch_url, "/catalog", cat_after, cs2) && cs2 == 200 &&
                              cat_after.is_array() && !cat_after.empty();

    cleanup();

    const bool ok = cat_ok_before && cat_ok_after && nodes_back && model_back;
    if (ok) {
        printf("test-cluster-e2e-orchestrator-restart: PASS (catalog persisted, nodes reconnected, model restored)\n");
        return 0;
    }
    fprintf(stderr, "test-cluster-e2e-orchestrator-restart: FAIL (catalog_before=%d catalog_after=%d nodes_back=%d model_back=%d)\n",
            cat_ok_before, cat_ok_after, nodes_back, model_back);
    return 1;
}

#endif // _WIN32
