// Task 7.5 - scenario: losing a node makes session creation fail with HTTP 503.
//
// Starts orchestrator + 3 nodes, kills node-b after registration, then expects
// POST /session/create to return 503 with a meaningful error message.

#include "e2e_common.h"

#include <string>

using e2e::json;

#if defined(_WIN32)
int main() {
    fprintf(stderr, "test-cluster-e2e-node-loss requires fork() (Unix)\n");
    return 1;
}
#else

int main(int argc, char ** argv) {
    const std::string model_id = "llama-3.2-1b";
    const std::string repository = "hugging-quants/Llama-3.2-1B-Instruct-Q4_K_M-GGUF";
    const std::string dir  = e2e::exe_dir(argv[0]);
    const std::string logs = e2e::log_dir() + "/node-loss";
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
    const int port_b    = base + 2;
    const int port_c    = base + 3;
    const std::string orch_url = "http://127.0.0.1:" + std::to_string(orch_port);
    const std::string store = std::filesystem::absolute(logs + "/store").string();

    pid_t pid_orch = 0, pid_a = 0, pid_b = 0, pid_c = 0;
    auto cleanup = [&]() {
        e2e::kill_wait(pid_a); e2e::kill_wait(pid_b);
        e2e::kill_wait(pid_c); e2e::kill_wait(pid_orch);
    };

    pid_orch = e2e::spawn_logged({
        dir + "orchestrator", "--model", gguf_abs,
        "--listen", "127.0.0.1:" + std::to_string(orch_port),
    }, logs + "/orchestrator.log");

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
    e2e::wait_http_ok("http://127.0.0.1:" + std::to_string(port_b), 200);

    // Task 9.1: session creation now requires the model to be registered.
    std::string reg_err;
    if (!e2e::register_model(orch_url, model_id, gguf_abs, reg_err, repository)) {
        fprintf(stderr, "FAIL: model registration: %s\n", reg_err.c_str());
        cleanup();
        return 1;
    }

    // Task 9.2: discover the model remotely before using it.
    std::string disc_err;
    if (!e2e::discover_model(orch_url, model_id, disc_err, 1)) {
        fprintf(stderr, "FAIL: model discovery: %s\n", disc_err.c_str());
        cleanup();
        return 1;
    }

    std::string manifest_err;
    if (!e2e::build_manifest(orch_url, model_id, manifest_err)) {
        fprintf(stderr, "FAIL: manifest build: %s\n", manifest_err.c_str());
        cleanup();
        return 1;
    }

    // Kill node-b and give the orchestrator a moment.
    e2e::kill_wait(pid_b);
    pid_b = 0;
    sleep(1);

    json out; int status = 0;
    e2e::http_post(orch_url, "/session/create", json({ { "model", model_id } }), out, status, 30);

    const std::string error = out.value("error", "");
    cleanup();

    if (status == 503 && !error.empty()) {
        printf("test-cluster-e2e-node-loss: PASS (503 with error: %s)\n", error.c_str());
        return 0;
    }
    fprintf(stderr, "test-cluster-e2e-node-loss: FAIL (expected 503 with message, got status=%d error='%s')\n",
            status, error.c_str());
    return 1;
}

#endif // _WIN32
