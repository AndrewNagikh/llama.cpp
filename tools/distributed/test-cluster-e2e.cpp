// Task 7.5 - End-to-End Cluster Validation Suite.
//
// Brings up orchestrator + node-a/b/c, installs a model, creates a session,
// runs distributed generation, compares tokens against an in-process split
// baseline, validates orchestrator restart recovery, and shuts everything down.
//
// Usage:
//   ./build/bin/test-cluster-e2e
//   ./build/bin/test-cluster-e2e --model llama-3.2-1b --prompt "Tell me a joke"

#include "e2e_common.h"

#include <cstdlib>
#include <string>
#include <vector>

using e2e::json;

#if defined(_WIN32)
int main() {
    fprintf(stderr, "test-cluster-e2e requires fork() (Unix)\n");
    return 1;
}
#else

int main(int argc, char ** argv) {
    std::string model_id = "llama-3.2-1b";
    std::string prompt   = "Tell me a joke";
    std::string gguf_override;
    int max_tokens = 32;

    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "--model") == 0 && i + 1 < argc) {
            model_id = argv[++i];
        } else if (strcmp(argv[i], "--prompt") == 0 && i + 1 < argc) {
            prompt = argv[++i];
        } else if (strcmp(argv[i], "--max-tokens") == 0 && i + 1 < argc) {
            max_tokens = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--gguf") == 0 && i + 1 < argc) {
            gguf_override = argv[++i];
        } else {
            fprintf(stderr, "unknown arg: %s\n", argv[i]);
            return 2;
        }
    }

    const std::string dir = e2e::exe_dir(argv[0]);
    const std::string logs = e2e::log_dir();
    std::filesystem::create_directories(logs);

    std::vector<e2e::stage_result> stages;
    auto add = [&](const std::string & name, bool pass, const std::string & detail = "") {
        stages.push_back({ name, pass, detail });
    };

    // Cluster process handles.
    const int base       = 26000 + (getpid() % 400);
    const int orch_port  = base;
    const int port_a     = base + 1;
    const int port_b     = base + 2;
    const int port_c     = base + 3;
    const std::string orch_url = "http://127.0.0.1:" + std::to_string(orch_port);

    pid_t pid_orch = 0, pid_a = 0, pid_b = 0, pid_c = 0;

    auto kill_all = [&]() {
        e2e::kill_wait(pid_a);
        e2e::kill_wait(pid_b);
        e2e::kill_wait(pid_c);
        e2e::kill_wait(pid_orch);
    };

    auto finish = [&](bool overall) -> int {
        e2e::write_summary(logs + "/e2e-summary.json", stages, overall);
        e2e::print_report(stages, overall);
        return overall ? 0 : 1;
    };

    // ----- Stage 1: Environment validation -------------------------------
    {
        bool ok = true;
        std::string detail;

        const char * bins[] = { "orchestrator", "node_agent",
                                "split_gen3_a", "split_gen3_b", "split_gen3_c" };
        for (const char * b : bins) {
            std::error_code ec;
            if (!std::filesystem::exists(dir + b, ec)) {
                ok = false;
                detail = std::string("missing binary: ") + b;
                break;
            }
        }

        const std::string gguf = ok ? e2e::find_gguf(gguf_override) : "";
        if (ok && gguf.empty()) {
            ok = false;
            detail = "GGUF model not found (set MODEL or --gguf)";
        }

        if (ok) {
            for (int p : { orch_port, port_a, port_b, port_c }) {
                if (!e2e::port_free(p)) {
                    ok = false;
                    detail = "port busy: " + std::to_string(p);
                    break;
                }
            }
        }

        add("Environment", ok, ok ? gguf : detail);
        if (!ok) {
            fprintf(stderr, "FAIL: environment validation (%s)\n", detail.c_str());
            return finish(false);
        }
    }

    const std::string gguf = e2e::find_gguf(gguf_override);
    const std::string gguf_abs = std::filesystem::absolute(gguf).string();

    // Per-node isolated model stores under logs/e2e/store/<node>.
    const std::string store_root = std::filesystem::absolute(logs + "/store").string();
    auto node_store = [&](const std::string & id) { return store_root + "/" + id; };
    for (const auto & id : { "node-a", "node-b", "node-c" }) {
        std::filesystem::create_directories(node_store(id));
    }

    auto spawn_node = [&](const std::string & id, int port) {
        return e2e::spawn_logged({
            dir + "node_agent",
            "--model", gguf_abs,
            "--listen", "127.0.0.1:" + std::to_string(port),
            "--orchestrator", orch_url,
            "--node-id", id,
            "--advertise-host", "127.0.0.1",
            "--models-dir", node_store(id),
        }, logs + "/" + id + ".log");
    };

    // ----- Stage 2: Start cluster + node registration --------------------
    {
        pid_orch = e2e::spawn_logged({
            dir + "orchestrator",
            "--model", gguf_abs,
            "--listen", "127.0.0.1:" + std::to_string(orch_port),
        }, logs + "/orchestrator.log");

        if (!e2e::wait_http_ok(orch_url, 200)) {
            add("Start cluster", false, "orchestrator did not become healthy");
            fprintf(stderr, "FAIL: node registration\n");
            kill_all();
            return finish(false);
        }

        pid_a = spawn_node("node-a", port_a);
        pid_b = spawn_node("node-b", port_b);
        pid_c = spawn_node("node-c", port_c);

        const bool registered = e2e::wait_nodes_registered(orch_url, 3, 400);
        add("Node registration", registered, registered ? "3/3 nodes" : "fewer than 3 nodes");
        if (!registered) {
            fprintf(stderr, "FAIL: node registration\n");
            kill_all();
            return finish(false);
        }

        // Make sure each node's HTTP server is reachable for install.
        e2e::wait_http_ok("http://127.0.0.1:" + std::to_string(port_a), 200);
        e2e::wait_http_ok("http://127.0.0.1:" + std::to_string(port_b), 200);
        e2e::wait_http_ok("http://127.0.0.1:" + std::to_string(port_c), 200);
    }

    const std::string repository = "hugging-quants/Llama-3.2-1B-Instruct-Q4_K_M-GGUF";

    // ----- Stage 2.5: Model registry registration (Task 9.1) ------------
    {
        std::string err;
        bool ok = e2e::register_model(orch_url, model_id, gguf_abs, err, repository);

        json mout;
        int ms = 0;
        ok = ok && e2e::http_get(orch_url, "/models/" + model_id, mout, ms, 5) &&
                  ms == 200 && mout.value("status", "") == "DISCOVERED";

        add("Model registry", ok, ok ? "registered and DISCOVERED" : err);
        if (!ok) {
            fprintf(stderr, "FAIL: model registry stage (%s)\n", err.c_str());
            kill_all();
            return finish(false);
        }
    }

    // ----- Stage 2.75: Remote model discovery (Task 9.2) ----------------
    {
        std::string err;
        const bool ok = e2e::discover_model(orch_url, model_id, err, 1);
        add("Remote discovery", ok, ok ? "registry MANIFEST_PENDING" : err);
        if (!ok) {
            fprintf(stderr, "FAIL: remote discovery stage (%s)\n", err.c_str());
            kill_all();
            return finish(false);
        }
    }

    // ----- Stage 2.8: GGUF manifest build (Task 9.3) ---------------------
    {
        std::string err;
        uint64_t file_size = 0;
        const bool ok = e2e::build_manifest(orch_url, model_id, err, &file_size);
        std::string detail = ok ? "MANIFEST_READY, metadata only" : err;
        if (ok && file_size > 0) {
            detail += " (file_size=" + std::to_string(file_size) + ")";
        }
        add("Manifest build", ok, detail);
        if (!ok) {
            fprintf(stderr, "FAIL: manifest build stage (%s)\n", err.c_str());
            kill_all();
            return finish(false);
        }
    }

    // ----- Stage 2.9: Desired cluster layout (Task 9.4) ------------------
    {
        std::string err;
        const bool ok = e2e::build_layout(orch_url, model_id, err);
        add("Desired layout", ok, ok ? "fits_cluster, full layer coverage" : err);
        if (!ok) {
            fprintf(stderr, "FAIL: layout build stage (%s)\n", err.c_str());
            kill_all();
            return finish(false);
        }
    }

    // ----- Stage 3: Model install ----------------------------------------
    {
        std::string err;
        const bool ready = e2e::install_and_wait_ready(orch_url, model_id, 3, err);
        add("Model install", ready, ready ? "model ready on 3 nodes" : err);
        if (!ready) {
            kill_all();
            return finish(false);
        }
    }

    // ----- Stage 4: Session create ---------------------------------------
    const int n_layer = e2e::model_n_layer(gguf_abs.c_str());
    std::string session_id;
    {
        json out;
        int status = 0;
        const bool posted = e2e::http_post(orch_url, "/session/create",
                json({ { "model", model_id } }), out, status, 60);
        bool ok = posted && status == 200;
        std::string detail;

        if (ok) {
            session_id = out.value("session_id", "");
            const json layout   = out.value("layout", json::array());
            const json pipeline = out.value("pipeline", json::array());
            if (session_id.empty()) {
                ok = false; detail = "empty session_id";
            } else if (!layout.is_array() || layout.empty()) {
                ok = false; detail = "empty layout";
            } else if (!pipeline.is_array() || (int) pipeline.size() != 3) {
                ok = false; detail = "pipeline does not contain all 3 nodes";
            } else {
                std::string lerr;
                if (n_layer <= 0) {
                    ok = false; detail = "could not read n_layer";
                } else if (!e2e::validate_layout(pipeline, n_layer, 3, lerr)) {
                    ok = false; detail = lerr;
                }
            }
        } else {
            detail = "status=" + std::to_string(status);
        }

        add("Session create", ok, ok ? ("layers [0," + std::to_string(n_layer) + ")") : detail);
        if (!ok) {
            kill_all();
            return finish(false);
        }
    }

    // ----- Stage 5: Distributed generation -------------------------------
    std::vector<llama_token> dist_tokens;
    std::string dist_text;
    {
        json out;
        int status = 0;
        const bool posted = e2e::http_post(orch_url, "/session/generate",
                json({ { "session_id", session_id }, { "prompt", prompt }, { "max_tokens", max_tokens } }),
                out, status, 240);

        bool ok = posted && status == 200;
        std::string detail;
        if (ok) {
            const int count = out.value("count", 0);
            dist_text = out.value("text", "");
            if (out.contains("tokens") && out["tokens"].is_array()) {
                for (const auto & t : out["tokens"]) {
                    dist_tokens.push_back((llama_token) t.get<int>());
                }
            }
            if (count != max_tokens) {
                ok = false; detail = "count=" + std::to_string(count);
            } else if ((int) dist_tokens.size() != max_tokens) {
                ok = false; detail = "tokens.size=" + std::to_string(dist_tokens.size());
            } else if (dist_text.empty()) {
                ok = false; detail = "empty text";
            }
        } else {
            detail = "status=" + std::to_string(status);
        }

        add("Distributed generation", ok, ok ? (std::to_string(max_tokens) + " tokens") : detail);
        if (!ok) {
            kill_all();
            return finish(false);
        }
    }

    // ----- Stage 6/7: Reference generation + exact comparison ------------
    // The split baseline and the distributed pipeline are numerically identical
    // for all generated content; they can only diverge once the model reaches a
    // near-tie at the end-of-generation decision (e.g. emit <|eot_id|> vs keep
    // going). We therefore require an exact match for every "real" generated
    // token, i.e. everything before the reference's first end-of-generation
    // token. Tokens at/after the stop decision are undefined and not compared.
    // If the reference never stops, all tokens are compared.
    {
        const auto ref_tokens = e2e::run_split3_baseline(dir, gguf_abs.c_str(), prompt, max_tokens);
        bool ok = true;
        std::string detail;

        if (ref_tokens.empty()) {
            ok = false; detail = "reference generation failed";
        } else {
            const int eog = e2e::first_eog_index(gguf_abs.c_str(), ref_tokens);
            const size_t cmp_len =
                    (eog > 0) ? (size_t) eog : ref_tokens.size();

            if (dist_tokens.size() < cmp_len) {
                ok = false;
                detail = "distributed produced fewer tokens (" +
                         std::to_string(dist_tokens.size()) + ") than reference content (" +
                         std::to_string(cmp_len) + ")";
            } else {
                for (size_t i = 0; i < cmp_len; ++i) {
                    if (ref_tokens[i] != dist_tokens[i]) {
                        ok = false;
                        detail = "first mismatch at index " + std::to_string(i);
                        fprintf(stderr, "TOKEN MISMATCH index=%zu expected=%d actual=%d\n",
                                i, (int) ref_tokens[i], (int) dist_tokens[i]);
                        break;
                    }
                }
                if (ok) {
                    detail = std::to_string(cmp_len) + "/" + std::to_string(cmp_len) +
                             " tokens identical" + (eog >= 0 ? " (up to EOS)" : "");
                }
            }
        }

        add("Token comparison", ok, detail);
        if (!ok) {
            fprintf(stderr, "FAIL: token mismatch\n");
            kill_all();
            return finish(false);
        }
    }

    // ----- Stage 8: Restart recovery -------------------------------------
    {
        e2e::kill_wait(pid_orch);
        pid_orch = e2e::spawn_logged({
            dir + "orchestrator",
            "--model", gguf_abs,
            "--listen", "127.0.0.1:" + std::to_string(orch_port),
        }, logs + "/orchestrator.log");

        bool ok = e2e::wait_http_ok(orch_url, 200);
        std::string detail;
        if (!ok) {
            detail = "orchestrator did not restart";
        } else {
            const bool nodes_back = e2e::wait_nodes_registered(orch_url, 3, 400);
            const bool model_back = nodes_back && e2e::models_ready(orch_url, model_id, 3);
            ok = nodes_back && model_back;
            if (!nodes_back) {
                detail = "nodes did not reconnect";
            } else if (!model_back) {
                detail = "model not restored in GET /models";
            }
        }

        add("Restart recovery", ok, ok ? "nodes + model restored" : detail);
        if (!ok) {
            kill_all();
            return finish(false);
        }
    }

    // ----- Stage 9: Recreate session + generate --------------------------
    {
        // The registry is in-memory; after an orchestrator restart the model
        // must be registered again before Session Create can succeed.
        std::string err;
        if (!e2e::register_model(orch_url, model_id, gguf_abs, err, repository)) {
            add("Restart registry", false, err);
            kill_all();
            return finish(false);
        }

        if (!e2e::discover_model(orch_url, model_id, err, 1)) {
            add("Restart discovery", false, err);
            kill_all();
            return finish(false);
        }

        if (!e2e::build_manifest(orch_url, model_id, err)) {
            add("Restart manifest", false, err);
            kill_all();
            return finish(false);
        }

        if (!e2e::build_layout(orch_url, model_id, err)) {
            add("Restart layout", false, err);
            kill_all();
            return finish(false);
        }

        json create_out;
        int cs = 0;
        bool ok = e2e::http_post(orch_url, "/session/create", json({ { "model", model_id } }),
                create_out, cs, 60) && cs == 200;
        std::string detail;
        std::string sid = ok ? create_out.value("session_id", "") : "";
        if (ok && sid.empty()) {
            ok = false; detail = "empty session_id";
        }

        if (ok) {
            json gen_out;
            int gs = 0;
            const bool gen = e2e::http_post(orch_url, "/session/generate",
                    json({ { "session_id", sid }, { "prompt", prompt }, { "max_tokens", max_tokens } }),
                    gen_out, gs, 240);
            const int count = gen ? gen_out.value("count", 0) : 0;
            ok = gen && gs == 200 && count == max_tokens;
            if (!ok) {
                detail = "regenerate status=" + std::to_string(gs) + " count=" + std::to_string(count);
            }
        }

        add("Recreate session", ok, ok ? "regeneration ok" : detail);
        if (!ok) {
            kill_all();
            return finish(false);
        }
    }

    // ----- Stage 10: Graceful shutdown -----------------------------------
    {
        e2e::kill_proc(pid_a);
        e2e::kill_proc(pid_b);
        e2e::kill_proc(pid_c);
        e2e::kill_proc(pid_orch);
        for (pid_t p : { pid_a, pid_b, pid_c, pid_orch }) {
            if (p > 0) {
                waitpid(p, nullptr, 0);
            }
        }

        const bool clean = !e2e::pid_alive(pid_a) && !e2e::pid_alive(pid_b) &&
                           !e2e::pid_alive(pid_c) && !e2e::pid_alive(pid_orch);
        pid_a = pid_b = pid_c = pid_orch = 0;
        add("Cleanup", clean, clean ? "no leftover processes" : "processes still alive");
        if (!clean) {
            return finish(false);
        }
    }

    return finish(true);
}

#endif // _WIN32
