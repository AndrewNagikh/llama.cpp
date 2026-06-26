// Task 9.8.1 - E2E verification: sync → verify PASS → delete blob → verify FAIL → repair → verify PASS.
//
// Uses a single-node cluster so the orchestrator models-dir matches the synced layer store.

#include "e2e_common.h"

#include <cstdlib>
#include <string>
#include <vector>

using e2e::json;

#if defined(_WIN32)
int main() {
    fprintf(stderr, "test-cluster-e2e-verification requires fork() (Unix)\n");
    return 1;
}
#else

int main(int argc, char ** argv) {
    std::string model_id = "llama-3.2-1b";
    std::string gguf_override;

    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "--model") == 0 && i + 1 < argc) {
            model_id = argv[++i];
        } else if (strcmp(argv[i], "--gguf") == 0 && i + 1 < argc) {
            gguf_override = argv[++i];
        } else {
            fprintf(stderr, "unknown arg: %s\n", argv[i]);
            return 2;
        }
    }

    const std::string dir  = e2e::exe_dir(argv[0]);
    const std::string logs = e2e::log_dir();
    std::filesystem::create_directories(logs);

    std::vector<e2e::stage_result> stages;
    auto add = [&](const std::string & name, bool pass, const std::string & detail = "") {
        stages.push_back({ name, pass, detail });
    };

    const int base      = 27000 + (getpid() % 400);
    const int orch_port = base;
    const int port_a    = base + 1;
    const std::string orch_url = "http://127.0.0.1:" + std::to_string(orch_port);

    pid_t pid_orch = 0;
    pid_t pid_a    = 0;

    auto kill_all = [&]() {
        e2e::kill_wait(pid_a);
        e2e::kill_wait(pid_orch);
    };

    auto finish = [&](bool overall) -> int {
        e2e::write_summary(logs + "/e2e-verification-summary.json", stages, overall);
        e2e::print_report(stages, overall);
        return overall ? 0 : 1;
    };

    const std::string gguf = e2e::find_gguf(gguf_override);
    if (gguf.empty()) {
        add("Environment", false, "GGUF model not found");
        return finish(false);
    }
    const std::string gguf_abs = std::filesystem::absolute(gguf).string();

    const std::string store_root = std::filesystem::absolute(logs + "/verify-store").string();
    const std::string node_store = store_root + "/node-a";
    std::filesystem::create_directories(node_store);

    // ----- Start orchestrator + single node --------------------------------
    {
        pid_orch = e2e::spawn_logged({
            dir + "orchestrator",
            "--model", gguf_abs,
            "--listen", "127.0.0.1:" + std::to_string(orch_port),
            "--models-dir", node_store,
        }, logs + "/orchestrator.log");

        if (!e2e::wait_http_ok(orch_url, 200)) {
            add("Start cluster", false, "orchestrator unhealthy");
            kill_all();
            return finish(false);
        }

        pid_a = e2e::spawn_logged({
            dir + "node_agent",
            "--model", gguf_abs,
            "--listen", "127.0.0.1:" + std::to_string(port_a),
            "--orchestrator", orch_url,
            "--node-id", "node-a",
            "--advertise-host", "127.0.0.1",
            "--models-dir", node_store,
            "--verify-materialization",
        }, logs + "/node-a.log");

        if (!e2e::wait_nodes_registered(orch_url, 1, 200)) {
            add("Node registration", false, "node-a not registered");
            kill_all();
            return finish(false);
        }
        add("Node registration", true, "1/1 nodes");
    }

    const std::string repository = "hugging-quants/Llama-3.2-1B-Instruct-Q4_K_M-GGUF";

    // ----- Register → discover → manifest → layout → sync ----------------
    {
        std::string err;
        bool ok = e2e::register_model(orch_url, model_id, gguf_abs, err, repository);
        ok = ok && e2e::discover_model(orch_url, model_id, err, 1);
        ok = ok && e2e::build_manifest(orch_url, model_id, err);
        ok = ok && e2e::build_layout(orch_url, model_id, err);
        ok = ok && e2e::sync_model_layers(orch_url, model_id, err);
        add("Synchronization", ok, ok ? "coverage READY" : err);
        if (!ok) {
            kill_all();
            return finish(false);
        }
    }

    // ----- Verification PASS ------------------------------------------------
    {
        std::string err;
        json report;
        const bool ok = e2e::run_model_verification(orch_url, model_id, err, &report, true);
        std::string detail = ok ? "all checks OK" : err;
        if (ok) {
            detail += " logits=" + report.value("logits", "?") +
                      " sampling=" + report.value("sampling", "?");
        }
        add("Verification", ok, detail);
        if (!ok) {
            kill_all();
            return finish(false);
        }
    }

    // ----- Delete blob → Verification FAIL → repair → PASS ------------------
    {
        std::string err;
        bool ok = true;

        const std::string blob_dir = node_store + "/" + model_id + "/layers";
        std::error_code ec;
        bool deleted = false;
        if (std::filesystem::exists(blob_dir, ec)) {
            for (const auto & ent : std::filesystem::directory_iterator(blob_dir, ec)) {
                if (ent.path().extension() == ".bin") {
                    std::filesystem::remove(ent.path(), ec);
                    deleted = true;
                    break;
                }
            }
        }
        if (!deleted) {
            ok = false;
            err = "no layer blob found to delete";
        }

        if (ok) {
            ok = e2e::run_model_verification(orch_url, model_id, err, nullptr, false);
        }
        if (ok) {
            ok = e2e::refresh_coverage(orch_url, model_id, err, "PARTIAL");
        }
        if (ok) {
            ok = e2e::sync_model_layers(orch_url, model_id, err);
        }
        if (ok) {
            ok = e2e::run_model_verification(orch_url, model_id, err, nullptr, true);
        }

        add("Verification repair", ok, ok ? "FAIL then PASS after repair" : err);
        if (!ok) {
            kill_all();
            return finish(false);
        }
    }

    kill_all();
    add("Shutdown", true, "clean");
    return finish(true);
}

#endif
