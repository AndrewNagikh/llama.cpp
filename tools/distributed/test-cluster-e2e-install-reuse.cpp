// Task 7.5 - scenario: repeated install must not re-download the model.
//
// Installs a model, then installs it again and verifies the node short-circuits
// to "already_installed" without re-fetching (installed_ms stays unchanged).

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

    // Register and discover the model before installing.
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
    std::string manifest_err;
    if (!e2e::build_manifest(orch_url, model_id, manifest_err)) {
        fprintf(stderr, "FAIL: manifest build: %s\n", manifest_err.c_str());
        cleanup();
        return 1;
    }
    std::string layout_err;
    if (!e2e::build_layout(orch_url, model_id, layout_err)) {
        fprintf(stderr, "FAIL: layout build: %s\n", layout_err.c_str());
        cleanup();
        return 1;
    }

    // First install -> ready.
    if (!e2e::install_and_wait_ready(orch_url, model_id, 1, err)) {
        fprintf(stderr, "FAIL: first install did not complete: %s\n", err.c_str());
        cleanup();
        return 1;
    }

    const std::string node_url = "http://127.0.0.1:" + std::to_string(port_a);
    auto read_installed_ms = [&]() -> uint64_t {
        json out; int s = 0;
        if (e2e::http_get(node_url, "/models/local", out, s) && s == 200 && out.is_array()) {
            for (const auto & m : out) {
                if (m.value("model_id", "") == model_id) {
                    return m.value("installed_ms", (uint64_t) 0);
                }
            }
        }
        return 0;
    };

    const uint64_t ms_before = read_installed_ms();

    // Second install -> should reuse, node reports already_installed.
    json install_out; int status = 0;
    if (!e2e::http_post(orch_url, "/models/install", json({ { "model", model_id } }),
            install_out, status, 30) || status != 200) {
        fprintf(stderr, "FAIL: second install request failed status=%d\n", status);
        cleanup();
        return 1;
    }
    const std::string job_id = install_out.value("job_id", "");

    // Poll the job and inspect per-node status.
    bool reused = false;
    for (int i = 0; i < 30; ++i) {
        json job; int js = 0;
        if (e2e::http_get(orch_url, "/models/install/" + job_id, job, js) && js == 200) {
            if (job.contains("nodes") && job["nodes"].is_object()) {
                for (auto it = job["nodes"].begin(); it != job["nodes"].end(); ++it) {
                    if (it.value().value("status", "") == "already_installed") {
                        reused = true;
                    }
                }
            }
            if (job.value("status", "") == "ready") {
                break;
            }
        }
        sleep(1);
    }

    const uint64_t ms_after = read_installed_ms();
    const bool unchanged = (ms_before != 0) && (ms_before == ms_after);

    cleanup();

    if (reused && unchanged) {
        printf("test-cluster-e2e-install-reuse: PASS (reused, installed_ms unchanged)\n");
        return 0;
    }
    fprintf(stderr, "test-cluster-e2e-install-reuse: FAIL (reused=%d unchanged=%d ms_before=%llu ms_after=%llu)\n",
            reused, unchanged, (unsigned long long) ms_before, (unsigned long long) ms_after);
    return 1;
}

#endif // _WIN32
