#include "dist_common.h"
#include "dist_http_fetch.h"
#include "dist_process.h"
#include "model_catalog.h"
#include "node_benchmark.h"
#include "node_agent/layer_store/layer_store.h"
#include "node_agent/layer_store/worker_builder.h"
#include "node_agent/layer_store/layer_gguf_assembler.h"
#include "architecture/semantic_runtime_descriptor.h"
#include "architecture/worker_requirement.h"
#include "runtime/runtime_config.h"
#include "runtime/runtime_role.h"
#include "runtime/runtime_worker_bind.h"
#include "runtime/runtime_worker_gguf_resolver.h"
#include "runtime/llama_layer_store_model_load.h"
#include "verification/worker_verify.h"
#include "node_agent/synchronization/synchronization_engine.h"
#include "orchestrator/install_planner/install_planner.h"
#include "orchestrator/coverage/coverage.h"
#include "orchestrator/manifest_builder/manifest_builder.h"

#include "httplib.h"
#include "nlohmann/json.hpp"
#include "llama.h"

#include "ggml-backend.h"
#include "transport/split_tcp_wire.h"
#include "transport/split_wave_wire.h"
#include "transport/runtime_protocol.h"
#include "transport/runtime_entry_queue.h"
#include "workers/split_gen_common.h"
#include "runtime_debug/runtime_debug.h"
#include "runtime_debug/perf_trace.h"
#include "runtime_debug/node_log_export.h"
#include "runtime_debug/perf_trace_export.h"
#include "runtime_debug/perf_gpu_sampler.h"
#include "runtime_debug/trace_recorder.h"

#include <algorithm>
#include <chrono>
#include <cctype>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <map>
#include <memory>
#include <mutex>
#include <set>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

using json = nlohmann::json;

// Task 19 level 2: lightweight network observability. Reuses the existing
// /health endpoint as a ping target (no new wire protocol) so a bench run
// can log RTT/jitter/loss to a peer alongside its tok/s and hit_rate,
// instead of guessing whether a throughput swing came from the wait
// policy or from the network drifting mid-run. Deliberately not a full
// heartbeat subsystem: one background thread, one endpoint.
struct peer_rtt_tracker {
    static constexpr size_t HISTORY = 128;
    std::mutex           mu;
    std::vector<double>  samples;
    size_t               next_slot = 0;
    size_t               attempts  = 0;
    size_t               successes = 0;

    void add_success(double rtt_ms) {
        std::lock_guard<std::mutex> lock(mu);
        ++attempts;
        ++successes;
        if (samples.size() < HISTORY) {
            samples.push_back(rtt_ms);
        } else {
            samples[next_slot] = rtt_ms;
            next_slot = (next_slot + 1) % HISTORY;
        }
    }

    void add_failure() {
        std::lock_guard<std::mutex> lock(mu);
        ++attempts;
    }

    json stats() {
        std::lock_guard<std::mutex> lock(mu);
        const double loss_pct = attempts > 0
                ? 100.0 * (double) (attempts - successes) / (double) attempts : 0.0;
        if (samples.empty()) {
            return {
                { "rtt_p50_ms", nullptr }, { "rtt_p95_ms", nullptr }, { "rtt_p99_ms", nullptr },
                { "jitter_ms", nullptr }, { "loss_pct", loss_pct }, { "samples", 0 },
            };
        }
        std::vector<double> sorted = samples;
        std::sort(sorted.begin(), sorted.end());
        const auto pct = [&](double p) { return sorted[(size_t) (p * (sorted.size() - 1))]; };
        double mean = 0.0;
        for (double v : sorted) { mean += v; }
        mean /= (double) sorted.size();
        double var = 0.0;
        for (double v : sorted) { var += (v - mean) * (v - mean); }
        var /= (double) sorted.size();
        return {
            { "rtt_p50_ms", pct(0.50) }, { "rtt_p95_ms", pct(0.95) }, { "rtt_p99_ms", pct(0.99) },
            { "jitter_ms", std::sqrt(var) }, { "loss_pct", loss_pct }, { "samples", sorted.size() },
        };
    }
};

static std::mutex g_peer_rtt_mu;
static std::map<std::string, std::unique_ptr<peer_rtt_tracker>> g_peer_rtt;

static std::string g_model_path;
static std::string g_node_id;
static int32_t g_n_layer = 0;
static int32_t g_n_embd  = 0;
static BenchmarkResult g_benchmark{};
static dist_child_process g_entry_worker{};
static dist_child_process g_middle_worker{};
static dist_child_process g_final_worker{};
static std::string g_entry_worker_state_file;
static std::string g_middle_worker_state_file;
static std::string g_final_worker_state_file;
static int g_pipeline_ctrl_port = 0;
static int g_pipeline_layer_end = 0;
static std::string g_pipeline_session_id;
static uint32_t g_pipeline_protocol = DIST_RUNTIME_PROTOCOL_V1;

struct node_runtime_stats {
    int configure_count        = 0;
    int materialization_count  = 0;
    int worker_spawn_count     = 0;
    int pipeline_generate_count = 0;
    int kv_cache_reset_count   = 0;
    int context_create_count   = 0;
    int runtime_load_count     = 0;
    int materialization_generation = 0;
    int tokenizer_init_count   = 0;
};

static node_runtime_stats g_runtime_stats;
static std::string g_entry_worker_gguf;
static std::string g_entry_model_id;
static constexpr const char * LAYER_STORE_MODEL_SENTINEL = "layer-store";

static std::string g_tokenizer_service_gguf;
static std::string g_tokenizer_service_model_id;
static std::string g_embedding_service_gguf;
static std::string g_embedding_service_model_id;
static int32_t g_embedding_service_layer_start = 0;
static int32_t g_embedding_service_layer_end   = 0;
static std::string g_output_service_gguf;
static std::string g_output_service_model_id;
static int32_t g_output_service_layer_start = 0;
static int32_t g_output_service_layer_end   = 0;
static bool g_sampler_service_ready = false;
static llama_model * g_entry_tokenizer = nullptr;
static llama_model * g_tokenizer_service_model = nullptr;
static llama_model * g_embedding_service_model = nullptr;
static llama_context * g_embedding_service_ctx = nullptr;
static int32_t g_embedding_service_n_embd = 0;
static bool g_external_embedding = false;
static std::string g_embedding_service_host;
static int g_embedding_service_port = 0;
static llama_model * g_output_service_model = nullptr;
static llama_context * g_output_service_ctx = nullptr;
static llama_sampler * g_output_service_smpl = nullptr;
static int32_t g_output_service_n_layer = 0;
static int32_t g_output_service_n_vocab = 0;
static bool g_external_output = false;
static std::string g_output_service_host;
static int g_output_service_port = 0;
static int g_agent_http_port = 0;
static std::string g_register_host;

static layer_store get_layer_store(const std::string & model_id);

static worker_role dist_role_to_worker_role(const dist_node_role role);

static void free_embedding_service() {
    if (g_embedding_service_ctx) {
        llama_free(g_embedding_service_ctx);
        g_embedding_service_ctx = nullptr;
    }
    if (g_embedding_service_model) {
        llama_model_free(g_embedding_service_model);
        g_embedding_service_model = nullptr;
    }
    g_embedding_service_n_embd = 0;
}

static bool ensure_embedding_service_loaded(std::string & err) {
    if (g_embedding_service_ctx != nullptr) {
        return true;
    }
    ggml_backend_load_all();

    if (g_embedding_service_gguf.empty() &&
            !g_embedding_service_model_id.empty() &&
            runtime_layer_first_enabled()) {
        layer_store store = get_layer_store(g_embedding_service_model_id);
        const auto manifest = store.load_manifest();
        if (!manifest.has_value()) {
            err = "embedding layer store manifest missing";
            return false;
        }
        runtime_layer_store_model_load_request req{};
        req.role        = worker_role::embedding;
        req.layer_start = g_embedding_service_layer_start;
        req.layer_end   = g_embedding_service_layer_end;
        g_embedding_service_model = runtime_load_model_from_layer_store(
                store, *manifest, req, err);
    } else {
        if (g_embedding_service_gguf.empty()) {
            err = "embedding service not configured";
            return false;
        }
        g_embedding_service_model = llama_model_load_from_file(
                g_embedding_service_gguf.c_str(), llama_model_default_params());
    }

    if (!g_embedding_service_model) {
        if (err.empty()) {
            err = "failed to load embedding model";
        }
        return false;
    }
    g_embedding_service_n_embd = llama_model_n_embd(g_embedding_service_model);
    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx   = 512;
    cparams.n_batch = 512;
    cparams.no_perf = true;
    g_embedding_service_ctx = llama_init_from_model(g_embedding_service_model, cparams);
    if (!g_embedding_service_ctx) {
        err = "failed to create embedding context";
        free_embedding_service();
        return false;
    }
    llama_set_layer_range(g_embedding_service_ctx, 0, 1);
    g_runtime_stats.runtime_load_count++;
    return true;
}

static bool embedding_service_compute_local(
        const std::vector<int32_t> & tokens,
        int32_t pos_start,
        std::vector<float> & hidden_out,
        int32_t & n_embd_out,
        std::string & err,
        bool clear_memory = false) {
    if (!ensure_embedding_service_loaded(err)) {
        return false;
    }
    if (clear_memory) {
        llama_memory_clear(llama_get_memory(g_embedding_service_ctx), true);
    }
    std::vector<llama_token> toks(tokens.begin(), tokens.end());
    n_embd_out = g_embedding_service_n_embd;
    hidden_out.resize((size_t) tokens.size() * (size_t) n_embd_out);
    if (tokens.size() > 1) {
        if (split_gen_decode_tokens(g_embedding_service_ctx, toks, pos_start, true) != 0) {
            err = "embedding forward failed";
            return false;
        }
        for (size_t i = 0; i < tokens.size(); ++i) {
            const float * h = llama_get_embeddings_ith(g_embedding_service_ctx, (int32_t) i);
            if (h == nullptr) {
                err = "embedding output missing";
                return false;
            }
            std::memcpy(
                    hidden_out.data() + i * (size_t) n_embd_out,
                    h,
                    (size_t) n_embd_out * sizeof(float));
        }
    } else if (split_gen_decode_one(g_embedding_service_ctx, toks[0], pos_start) != 0) {
        err = "embedding forward failed";
        return false;
    } else {
        const float * h = llama_get_embeddings(g_embedding_service_ctx);
        if (h == nullptr) {
            err = "embedding output missing";
            return false;
        }
        std::memcpy(hidden_out.data(), h, (size_t) n_embd_out * sizeof(float));
    }
    return true;
}

static bool embedding_service_reset(std::string & err) {
    const bool local = (!g_embedding_service_gguf.empty() || !g_embedding_service_model_id.empty()) &&
            (g_embedding_service_host.empty() ||
             g_embedding_service_host == g_register_host ||
             g_embedding_service_host == "127.0.0.1" ||
             g_embedding_service_host == "localhost");
    if (local) {
        if (g_embedding_service_ctx != nullptr) {
            llama_memory_clear(llama_get_memory(g_embedding_service_ctx), true);
            llama_clear_hidden_state(g_embedding_service_ctx);
        }
        return true;
    }
    if (g_embedding_service_host.empty() || g_embedding_service_port <= 0) {
        err = "embedding service endpoint not configured";
        return false;
    }
    httplib::Client cli(g_embedding_service_host.c_str(), g_embedding_service_port);
    cli.set_connection_timeout(10, 0);
    cli.set_read_timeout(60, 0);
    const auto res = cli.Post("/runtime/embedding/reset", "{}", "application/json");
    if (!res || res->status != 200) {
        err = "remote embedding service reset failed";
        return false;
    }
    return true;
}

static bool embedding_service_compute_remote(
        const std::vector<int32_t> & tokens,
        int32_t pos_start,
        std::vector<float> & hidden_out,
        int32_t & n_embd_out,
        std::string & err) {
    if (g_embedding_service_host.empty() || g_embedding_service_port <= 0) {
        err = "embedding service endpoint not configured";
        return false;
    }
    httplib::Client cli(g_embedding_service_host.c_str(), g_embedding_service_port);
    cli.set_connection_timeout(10, 0);
    cli.set_read_timeout(120, 0);
    json body = json::object();
    json toks = json::array();
    for (int32_t t : tokens) {
        toks.push_back(t);
    }
    body["tokens"]    = toks;
    body["pos_start"] = pos_start;
    const auto res = cli.Post("/runtime/embedding/embed", body.dump(), "application/json");
    if (!res || res->status != 200) {
        err = "remote embedding service failed";
        return false;
    }
    try {
        const json j = json::parse(res->body);
        if (!j.value("ok", false)) {
            err = j.value("error", "embed failed");
            return false;
        }
        n_embd_out = j.value("n_embd", 0);
        hidden_out.clear();
        for (const auto & v : j.at("hidden")) {
            hidden_out.push_back(v.get<float>());
        }
        if (n_embd_out <= 0 || hidden_out.empty()) {
            err = "invalid embedding response";
            return false;
        }
        return true;
    } catch (...) {
        err = "invalid embedding response json";
        return false;
    }
}

static bool embedding_service_compute(
        const std::vector<int32_t> & tokens,
        int32_t pos_start,
        std::vector<float> & hidden_out,
        int32_t & n_embd_out,
        std::string & err) {
    const bool local = (!g_embedding_service_gguf.empty() || !g_embedding_service_model_id.empty()) &&
            (g_embedding_service_host.empty() ||
             g_embedding_service_host == g_register_host ||
             g_embedding_service_host == "127.0.0.1" ||
             g_embedding_service_host == "localhost");
    if (local) {
        return embedding_service_compute_local(tokens, pos_start, hidden_out, n_embd_out, err);
    }
    return embedding_service_compute_remote(tokens, pos_start, hidden_out, n_embd_out, err);
}

static bool embedding_service_ready() {
    std::string err;
    if (!g_embedding_service_gguf.empty() || !g_embedding_service_model_id.empty()) {
        return ensure_embedding_service_loaded(err);
    }
    return !g_embedding_service_host.empty() && g_embedding_service_port > 0;
}

static void free_output_service() {
    if (g_output_service_smpl) {
        llama_sampler_free(g_output_service_smpl);
        g_output_service_smpl = nullptr;
    }
    if (g_output_service_ctx) {
        llama_free(g_output_service_ctx);
        g_output_service_ctx = nullptr;
    }
    if (g_output_service_model) {
        llama_model_free(g_output_service_model);
        g_output_service_model = nullptr;
    }
    g_output_service_n_layer = 0;
    g_output_service_n_vocab = 0;
}

static bool ensure_output_service_loaded(std::string & err) {
    if (g_output_service_ctx != nullptr) {
        return true;
    }
    ggml_backend_load_all();

    if (g_output_service_gguf.empty() &&
            !g_output_service_model_id.empty() &&
            runtime_layer_first_enabled()) {
        layer_store store = get_layer_store(g_output_service_model_id);
        const auto manifest = store.load_manifest();
        if (!manifest.has_value()) {
            err = "output layer store manifest missing";
            return false;
        }
        runtime_layer_store_model_load_request req{};
        req.role        = worker_role::output_head;
        req.layer_start = g_output_service_layer_start;
        req.layer_end   = g_output_service_layer_end;
        g_output_service_model = runtime_load_model_from_layer_store(
                store, *manifest, req, err);
    } else {
        if (g_output_service_gguf.empty()) {
            err = "output service not configured";
            return false;
        }
        g_output_service_model = llama_model_load_from_file(
                g_output_service_gguf.c_str(), llama_model_default_params());
    }

    if (!g_output_service_model) {
        if (err.empty()) {
            err = "failed to load output model";
        }
        return false;
    }
    g_output_service_n_layer = llama_model_n_layer(g_output_service_model);
    g_output_service_n_vocab = llama_vocab_n_tokens(
            llama_model_get_vocab(g_output_service_model));
    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx   = 512;
    cparams.n_batch = 512;
    cparams.no_perf = true;
    g_output_service_ctx = llama_init_from_model(g_output_service_model, cparams);
    if (!g_output_service_ctx) {
        err = "failed to create output context";
        free_output_service();
        return false;
    }
    llama_set_layer_range(g_output_service_ctx, g_output_service_n_layer - 1, g_output_service_n_layer);
    g_runtime_stats.runtime_load_count++;
    return true;
}

static bool output_service_sample_local(
        const float * hidden,
        int32_t n_embd,
        int32_t pos,
        int32_t & token_out,
        std::string & err) {
    if (!ensure_output_service_loaded(err)) {
        return false;
    }
    llama_memory_clear(llama_get_memory(g_output_service_ctx), true);
    llama_clear_hidden_state(g_output_service_ctx);
    if (split_gen_decode_hidden(
                g_output_service_ctx, hidden, 1, n_embd, pos, true, true) != 0) {
        err = "output forward failed";
        return false;
    }
    if (!g_output_service_smpl) {
        g_output_service_smpl = split_gen_make_sampler();
    }
    token_out = (int32_t) llama_sampler_sample(g_output_service_smpl, g_output_service_ctx, -1);
    llama_sampler_accept(g_output_service_smpl, (llama_token) token_out);
    return true;
}

static bool output_service_sample_remote(
        const float * hidden,
        int32_t n_embd,
        int32_t pos,
        int32_t & token_out,
        std::string & err) {
    if (g_output_service_host.empty() || g_output_service_port <= 0) {
        err = "output service endpoint not configured";
        return false;
    }
    httplib::Client cli(g_output_service_host.c_str(), g_output_service_port);
    cli.set_connection_timeout(10, 0);
    cli.set_read_timeout(120, 0);
    json hidden_json = json::array();
    for (int32_t i = 0; i < n_embd; ++i) {
        hidden_json.push_back(hidden[i]);
    }
    const json body = {
        { "hidden", hidden_json },
        { "n_embd", n_embd },
        { "pos", pos },
    };
    const auto res = cli.Post("/runtime/output/sample", body.dump(), "application/json");
    if (!res || res->status != 200) {
        err = "remote output service failed";
        return false;
    }
    try {
        const json j = json::parse(res->body);
        if (!j.value("ok", false)) {
            err = j.value("error", "sample failed");
            return false;
        }
        token_out = j.value("token_id", -1);
        if (token_out < 0) {
            err = "invalid output response";
            return false;
        }
        return true;
    } catch (...) {
        err = "invalid output response json";
        return false;
    }
}

static bool output_service_sample(
        const float * hidden,
        int32_t n_embd,
        int32_t pos,
        int32_t & token_out,
        std::string & err) {
    const bool local = !g_output_service_gguf.empty() &&
            (g_output_service_host.empty() ||
             g_output_service_host == g_register_host ||
             g_output_service_host == "127.0.0.1" ||
             g_output_service_host == "localhost");
    if (local) {
        return output_service_sample_local(hidden, n_embd, pos, token_out, err);
    }
    return output_service_sample_remote(hidden, n_embd, pos, token_out, err);
}

static void output_service_reset_local() {
    if (g_output_service_ctx != nullptr) {
        llama_memory_clear(llama_get_memory(g_output_service_ctx), true);
        llama_clear_hidden_state(g_output_service_ctx);
    }
    if (g_output_service_smpl != nullptr) {
        llama_sampler_free(g_output_service_smpl);
        g_output_service_smpl = nullptr;
    }
}

static void free_tokenizer_service() {
    if (g_tokenizer_service_model) {
        llama_model_free(g_tokenizer_service_model);
        g_tokenizer_service_model = nullptr;
    }
}

static void free_entry_tokenizer() {
    if (g_entry_tokenizer) {
        llama_model_free(g_entry_tokenizer);
        g_entry_tokenizer = nullptr;
    }
}

static llama_model_params default_tokenizer_model_params() {
    llama_model_params params = llama_model_default_params();
    params.vocab_only   = true;
    params.n_gpu_layers = 0;
    return params;
}

static bool apply_service_model_source(
        const std::string & worker_gguf,
        const std::string & model_id,
        std::string & gguf_out,
        std::string & model_id_out) {
    if (!worker_gguf.empty() && worker_gguf != LAYER_STORE_MODEL_SENTINEL) {
        gguf_out       = worker_gguf;
        model_id_out.clear();
        return true;
    }
    if (!model_id.empty() && runtime_layer_first_enabled()) {
        gguf_out.clear();
        model_id_out = model_id;
        return true;
    }
    return false;
}

static const llama_vocab * tokenizer_service_vocab() {
    if (!g_tokenizer_service_model) {
        if (!g_tokenizer_service_gguf.empty()) {
            ggml_backend_load_all();
            g_tokenizer_service_model = llama_model_load_from_file(
                    g_tokenizer_service_gguf.c_str(), default_tokenizer_model_params());
        } else if (!g_tokenizer_service_model_id.empty() && runtime_layer_first_enabled()) {
            layer_store store = get_layer_store(g_tokenizer_service_model_id);
            const auto manifest = store.load_manifest();
            std::string load_err;
            if (manifest.has_value()) {
                runtime_layer_store_model_load_request req{};
                req.role           = worker_role::tokenizer;
                req.params         = default_tokenizer_model_params();
                req.verify_tensors = false;
                g_tokenizer_service_model = runtime_load_model_from_layer_store(
                        store, *manifest, req, load_err);
            }
            if (!g_tokenizer_service_model && !load_err.empty()) {
                fprintf(stderr, "node_agent: tokenizer layer store load: %s\n", load_err.c_str());
            }
        }
        if (g_tokenizer_service_model) {
            g_runtime_stats.tokenizer_init_count = 1;
            if (!g_tokenizer_service_gguf.empty()) {
                fprintf(stderr,
                        "node_agent: tokenizer service loaded from %s\n",
                        g_tokenizer_service_gguf.c_str());
            } else {
                fprintf(stderr,
                        "node_agent: tokenizer service loaded from layer store %s\n",
                        g_tokenizer_service_model_id.c_str());
            }
        } else if (!g_tokenizer_service_gguf.empty()) {
            fprintf(stderr,
                    "node_agent: tokenizer load failed for %s\n",
                    g_tokenizer_service_gguf.c_str());
        }
    }
    return g_tokenizer_service_model ? llama_model_get_vocab(g_tokenizer_service_model) : nullptr;
}

static const llama_vocab * entry_node_vocab() {
    if (!g_entry_tokenizer) {
        if (!g_entry_worker_gguf.empty() &&
                g_entry_worker_gguf != LAYER_STORE_MODEL_SENTINEL) {
            ggml_backend_load_all();
            g_entry_tokenizer = llama_model_load_from_file(
                    g_entry_worker_gguf.c_str(), default_tokenizer_model_params());
        } else if (!g_entry_model_id.empty() && runtime_layer_first_enabled()) {
            layer_store store = get_layer_store(g_entry_model_id);
            const auto manifest = store.load_manifest();
            std::string load_err;
            if (manifest.has_value()) {
                runtime_layer_store_model_load_request req{};
                req.role           = worker_role::tokenizer;
                req.params         = default_tokenizer_model_params();
                req.verify_tensors = false;
                g_entry_tokenizer = runtime_load_model_from_layer_store(
                        store, *manifest, req, load_err);
            }
            if (!g_entry_tokenizer && !load_err.empty()) {
                fprintf(stderr, "node_agent: entry tokenizer layer store load: %s\n", load_err.c_str());
            }
        }
        if (g_entry_tokenizer) {
            g_runtime_stats.tokenizer_init_count = 1;
        }
    }
    return g_entry_tokenizer ? llama_model_get_vocab(g_entry_tokenizer) : nullptr;
}

static json node_runtime_stats_json() {
    return {
        { "configure_count", g_runtime_stats.configure_count },
        { "materialization_count", g_runtime_stats.materialization_count },
        { "worker_spawn_count", g_runtime_stats.worker_spawn_count },
        { "pipeline_generate_count", g_runtime_stats.pipeline_generate_count },
        { "kv_cache_reset_count", g_runtime_stats.kv_cache_reset_count },
        { "context_create_count", g_runtime_stats.context_create_count },
        { "runtime_load_count", g_runtime_stats.runtime_load_count },
        { "materialization_generation", g_runtime_stats.materialization_generation },
        { "tokenizer_init_count", g_runtime_stats.tokenizer_init_count },
        { "materialized", g_runtime_stats.materialization_count > 0 },
        { "runtime_loaded", g_runtime_stats.runtime_load_count > 0 },
    };
}
static model_store g_model_store;
static std::mutex g_model_store_mu;
static std::set<std::string> g_active_downloads;
static std::map<std::string, std::string> g_download_errors;
static std::string g_models_state_path;
static std::string g_models_dir;
static bool g_verify_materialization = false;
static synchronization_engine g_sync_engine;
static std::mutex g_layer_store_mu;

static layer_store get_layer_store(const std::string & model_id) {
    std::lock_guard<std::mutex> lock(g_layer_store_mu);
    layer_store store(g_model_store.get_models_dir(), model_id);
    return store;
}

static void perf_consume_trace_from_body(const json & body, const char * component) {
    const std::string trace_id = body.value("trace_id", "");
    const bool perf_trace = body.value("perf_trace", false);
    if (!perf_trace && !perf_trace_enabled()) {
        return;
    }
    if (perf_trace) {
        dist_set_env("DIST_PERF_TRACE", "1");
        if (!g_models_dir.empty()) {
            dist_set_env("DIST_PERF_TRACE_DIR", (g_models_dir + "/perf_trace").c_str());
        }
        // g_cfg is cached at first perf_trace_enabled() call; without this,
        // a long-lived node_agent that already cached enabled=false ignores
        // DIST_PERF_TRACE=1 above, and every worker this /configure spawns
        // inherits an environment with tracing off for its whole lifetime
        // (env vars set here after fork can never reach an already-running
        // child either way, so this must land before start_worker() runs).
        perf_trace_reload_config();
    }
    if (!trace_id.empty()) {
        perf_trace_set_node_id(g_node_id);
        perf_trace_set_component(component);
        perf_trace_set_context(trace_id, "session_create", -1);
    }
}

static uint64_t now_ms() {
    const auto now = std::chrono::system_clock::now();
    const auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(now.time_since_epoch());
    return static_cast<uint64_t>(ms.count());
}

static std::string shell_quote(const std::string & value) {
    std::string quoted = "'";
    for (char c : value) {
        if (c == '\'') {
            quoted += "'\\''";
        } else {
            quoted += c;
        }
    }
    quoted += "'";
    return quoted;
}

static uint64_t file_size_or_zero(const std::string & path) {
    std::error_code ec;
    const auto size = std::filesystem::file_size(path, ec);
    return ec ? 0 : static_cast<uint64_t>(size);
}

static void usage(const char * prog) {
    fprintf(stderr,
            "usage: %s --listen HOST:PORT --orchestrator URL "
            "[--model PATH] [--node-id ID] [--advertise-host IP] [--models-dir DIR] [--verify-materialization] [--rebenchmark] [--distributed-debug]\n"
            "  Layer-first mode: omit --model; workers load GGUF assembled from synced layers.\n"
            "example: %s --listen 0.0.0.0:9001 --orchestrator http://10.0.0.1:9000 --advertise-host 10.0.0.2\n",
            prog, prog);
}

static bool parse_args(int argc, char ** argv, std::string & listen, std::string & orchestrator,
        std::string & node_id, std::string & advertise_host, bool & rebenchmark) {
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "--model") == 0 && i + 1 < argc) {
            g_model_path = argv[++i];
        } else if (strcmp(argv[i], "--listen") == 0 && i + 1 < argc) {
            listen = argv[++i];
        } else if (strcmp(argv[i], "--orchestrator") == 0 && i + 1 < argc) {
            orchestrator = argv[++i];
        } else if (strcmp(argv[i], "--node-id") == 0 && i + 1 < argc) {
            node_id = argv[++i];
        } else if (strcmp(argv[i], "--advertise-host") == 0 && i + 1 < argc) {
            advertise_host = argv[++i];
        } else if (strcmp(argv[i], "--models-dir") == 0 && i + 1 < argc) {
            g_models_dir = argv[++i];
        } else if (strcmp(argv[i], "--verify-materialization") == 0) {
            g_verify_materialization = true;
        } else if (strcmp(argv[i], "--rebenchmark") == 0) {
            rebenchmark = true;
        } else if (strcmp(argv[i], "--distributed-debug") == 0) {
            dist_set_env("LLAMA_DISTRIBUTED_DEBUG", "1");
        } else {
            return false;
        }
    }
    if (g_models_dir.empty()) {
        if (const char * env = std::getenv("MODELS_DIR")) {
            g_models_dir = env;
        } else {
            g_models_dir = dist_home_dir() + "/.distributed-llm/models";
        }
    }
    return !listen.empty() && !orchestrator.empty();
}

static std::string exe_dir(const char * argv0) {
    std::string p(argv0);
    const auto pos = p.find_last_of("/\\");
    if (pos != std::string::npos) {
        return p.substr(0, pos + 1);
    }
    return "./";
}

// Locate an already-present GGUF that satisfies the requested catalog file,
// so install can seed locally (symlink) instead of downloading over the network.
static std::string find_local_source(const std::string & file) {
    if (file.empty()) {
        return "";
    }
    // The node already runs against a model file; reuse it when names match.
    if (!g_model_path.empty()) {
        std::error_code ec;
        if (std::filesystem::exists(g_model_path, ec)) {
            const std::string base = std::filesystem::path(g_model_path).filename().string();
            if (base == file) {
                return g_model_path;
            }
        }
    }
    return "";
}

static void start_model_download(
        std::string model_id,
        std::string repo,
        std::string file,
        std::string output_path,
        std::string downloader_script) {
    std::thread([model_id = std::move(model_id),
                 repo = std::move(repo),
                 file = std::move(file),
                 output_path = std::move(output_path),
                 downloader_script = std::move(downloader_script)]() {
        const std::string progress_path = output_path + ".progress.json";
        const std::string command =
                dist_python_cmd() + " " + shell_quote(downloader_script) +
                " --repo " + shell_quote(repo) +
                " --file " + shell_quote(file) +
                " --output " + shell_quote(output_path) +
                " --progress-file " + shell_quote(progress_path);

        const int rc = std::system(command.c_str());
        const bool ok = (rc == 0) && std::filesystem::exists(output_path) && file_size_or_zero(output_path) > 0;

        std::lock_guard<std::mutex> lock(g_model_store_mu);
        installed_model model;
        model.model_id = model_id;
        model.local_path = output_path;
        model.size_bytes = file_size_or_zero(output_path);
        model.ready = ok;
        model.installed_ms = ok ? now_ms() : 0;
        g_model_store.add_model(model);

        g_active_downloads.erase(model_id);
        if (ok) {
            g_download_errors.erase(model_id);
        } else {
            g_download_errors[model_id] = "download failed";
        }
        g_model_store.save_state(g_models_state_path);
    }).detach();
}

static bool load_model_metadata() {
    ggml_backend_load_all();
    llama_model * model = llama_model_load_from_file(g_model_path.c_str(), llama_model_default_params());
    if (!model) {
        return false;
    }
    g_n_layer = llama_model_n_layer(model);
    g_n_embd  = llama_model_n_embd(model);
    llama_model_free(model);
    return true;
}

static dist_configure_req parse_configure(const json & body) {
    dist_configure_req req{};
    const std::string role = body.value("role", "");
    if (role == "entry") {
        req.role = DIST_ROLE_ENTRY;
    } else if (role == "middle") {
        req.role = DIST_ROLE_MIDDLE;
    } else if (role == "final") {
        req.role = DIST_ROLE_FINAL;
    }
    req.layer_start = body.value("layer_start", 0);
    req.layer_end   = body.value("layer_end", 0);
    req.session_id  = body.value("session_id", "");
    req.model_id    = body.value("model_id", "");
    req.ctrl_port   = body.value("ctrl_port", 0);
    req.peer_port   = body.value("peer_port", 0);
    req.next_host   = body.value("next_host", "127.0.0.1");
    req.next_port   = body.value("next_port", 0);
    req.peer_bind   = body.value("peer_bind", "0.0.0.0");
    req.next_is_final = body.value("next_is_final", false);
    req.source_url    = body.value("source_url", "");
    if (body.contains("output_service") && body["output_service"].is_object()) {
        const auto & os = body["output_service"];
        req.output_service_host = os.value("host", "");
        req.output_service_port = os.value("port", 0);
    }
    req.fa_port     = body.value("fa_port", 0);
    req.fa_host     = body.value("fa_host", "127.0.0.1");
    req.draft_model = body.value("draft_model", "");
    req.draft_k     = body.value("draft_k", 4);
    return req;
}

static bool register_with_orchestrator(
        const std::string & orchestrator,
        const std::string & node_id,
        const std::string & host,
        int http_port) {
    httplib::Client cli(orchestrator.c_str());
    cli.set_connection_timeout(5, 0);
    cli.set_read_timeout(10, 0);

    const double score = g_benchmark.score;

    dist_node_memory memory{};
    dist_node_cpu cpu{};
    const dist_node_system sys = dist_probe_node_system();
    dist_probe_node_memory(memory);
    dist_probe_node_cpu(cpu);
    const auto caps = dist_probe_capabilities();

    const auto ram_gb  = static_cast<int>(memory.total_ram_bytes / (1024 * 1024 * 1024));
    const auto vram_gb = static_cast<int>(memory.total_vram_bytes / (1024 * 1024 * 1024));

    json body = {
        { "node_id", node_id },
        { "host", host },
        { "port", http_port },
        { "n_layer", g_n_layer },
        { "n_embd", g_n_embd },
        { "score", score },
        { "memory_total_mb", static_cast<int64_t>(memory.total_ram_bytes / (1024 * 1024)) },
        { "memory_free_mb", static_cast<int64_t>(memory.free_ram_bytes / (1024 * 1024)) },
        { "memory", {
            { "total_ram", memory.total_ram_bytes },
            { "free_ram", memory.free_ram_bytes },
            { "total_vram", memory.total_vram_bytes },
            { "free_vram", memory.free_vram_bytes },
            { "has_gpu", memory.has_gpu },
        }},
        { "hardware", {
            { "backend", caps.gpu_backend },
            { "gpu_name", caps.gpu_name },
            { "cpu_name", cpu.cpu_name },
            { "logical_cores", cpu.logical_cores },
            { "physical_cores", cpu.physical_cores },
            { "cpu_threads", caps.cpu_threads },
            { "ram_gb", ram_gb },
            { "gpu_vram_gb", vram_gb },
        }},
        { "capabilities", {
            { "gpu_backend", caps.gpu_backend },
            { "gpu_memory_mb", caps.gpu_memory_mb },
            { "cpu_threads", caps.cpu_threads },
            { "supported_arch", caps.supported_arch },
        }},
        { "system", {
            { "os", sys.os },
            { "arch", sys.arch },
        }},
        { "performance", {
            { "score", score },
            { "decode_tps", g_benchmark.decode_tps },
            { "prefill_tps", g_benchmark.prefill_tps },
            { "load_ms", g_benchmark.load_ms },
        }},
    };

    const auto res = cli.Post("/register", body.dump(), "application/json");
    if (!res || res->status != 200) {
        if (!res) {
            fprintf(stderr, "node_agent: register failed — cannot reach orchestrator at %s\n",
                    orchestrator.c_str());
        } else {
            fprintf(stderr, "node_agent: register failed HTTP %d: %s\n",
                    res->status, res->body.c_str());
        }
        return false;
    }

    fprintf(stderr, "node_agent: registered as %s at %s:%d score=%.1f\n",
            node_id.c_str(), host.c_str(), http_port, score);
    return true;
}

static dist_child_process * worker_proc_slot(const dist_node_role role) {
    switch (role) {
        case DIST_ROLE_ENTRY:  return &g_entry_worker;
        case DIST_ROLE_MIDDLE: return &g_middle_worker;
        case DIST_ROLE_FINAL:  return &g_final_worker;
        default:               return nullptr;
    }
}

static std::string * worker_state_file_slot(const dist_node_role role) {
    switch (role) {
        case DIST_ROLE_ENTRY:  return &g_entry_worker_state_file;
        case DIST_ROLE_MIDDLE: return &g_middle_worker_state_file;
        case DIST_ROLE_FINAL:  return &g_final_worker_state_file;
        default:               return nullptr;
    }
}

static std::string worker_ready_state_file(
        const dist_node_role role,
        const std::string & session_id) {
    std::string base = g_models_dir.empty() ? "/tmp" : g_models_dir;
    std::string sid = session_id.empty() ? "default" : session_id;
    for (char & c : sid) {
        if (!(std::isalnum((unsigned char) c) || c == '-' || c == '_')) {
            c = '_';
        }
    }
    return base + "/worker-state-" + g_node_id + "-" + dist_role_name(role) + "-" + sid + ".txt";
}

static std::string read_worker_ready_state_file(const std::string & path) {
    std::ifstream in(path);
    std::string state;
    if (in.good()) {
        std::getline(in, state);
    }
    return state.empty() ? "STARTING" : state;
}

static std::string worker_ready_state_for_role(const dist_node_role role) {
    std::string * state_file = worker_state_file_slot(role);
    const std::string state = read_worker_ready_state_file(state_file ? *state_file : "");
    dist_child_process * proc = worker_proc_slot(role);
    if (proc == nullptr || proc->pid == 0) {
        return state == "STARTING" ? state : "FAILED";
    }
    if (!dist_process_is_running(*proc)) {
        if (state_file != nullptr && !state_file->empty()) {
            split_gen_write_ready_state(*state_file, "FAILED");
        }
        return "FAILED";
    }
    return state;
}

static int worker_ready_state_rank(const std::string & state) {
    if (state == "FAILED") {
        return -1;
    }
    if (state == "STARTING") {
        return 0;
    }
    if (state == "MODEL_LOADING") {
        return 1;
    }
    if (state == "LISTENER_READY") {
        return 2;
    }
    if (state == "PIPE_READY") {
        return 3;
    }
    if (state == "READY") {
        return 4;
    }
    return 0;
}

static json worker_ready_states_json() {
    return {
        { "entry", worker_ready_state_for_role(DIST_ROLE_ENTRY) },
        { "middle", worker_ready_state_for_role(DIST_ROLE_MIDDLE) },
        { "final", worker_ready_state_for_role(DIST_ROLE_FINAL) },
    };
}

static bool wait_worker_ready_state(
        const dist_node_role role,
        const std::string & state_file,
        const std::string & target_state,
        const int timeout_ms,
        std::string & final_state,
        std::string & err) {
    const int target_rank = worker_ready_state_rank(target_state);
    const auto start = std::chrono::steady_clock::now();
    while (true) {
        final_state = read_worker_ready_state_file(state_file);
        if (worker_ready_state_rank(final_state) >= target_rank) {
            return true;
        }
        if (final_state == "FAILED") {
            err = std::string(dist_role_name(role)) + " worker failed during startup";
            return false;
        }
        const auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::steady_clock::now() - start).count();
        if (elapsed >= timeout_ms) {
            err = std::string(dist_role_name(role)) + " worker readiness timeout waiting for " +
                  target_state + " at state " + final_state;
            return false;
        }
        dist_sleep_ms(100);
    }
}

static void stop_worker_for_role(const dist_node_role role) {
    dist_child_process * slot = worker_proc_slot(role);
    if (!slot || slot->pid == 0) {
        return;
    }
    dist_process_kill(*slot);
    dist_process_reap(*slot);
    *slot = {};
    if (std::string * state_slot = worker_state_file_slot(role)) {
        if (!state_slot->empty()) {
            std::remove(state_slot->c_str());
            state_slot->clear();
        }
    }
    if (role == DIST_ROLE_ENTRY) {
        g_pipeline_ctrl_port = 0;
        g_pipeline_layer_end = 0;
    }
}

static void stop_all_workers() {
    stop_worker_for_role(DIST_ROLE_ENTRY);
    stop_worker_for_role(DIST_ROLE_MIDDLE);
    stop_worker_for_role(DIST_ROLE_FINAL);
}

static bool pipeline_connect_and_negotiate(int & ctrl_fd, std::string & err) {
    ctrl_fd = split_tcp_connect_retry("127.0.0.1", g_pipeline_ctrl_port, 300, 100);
    if (ctrl_fd < 0) {
        err = "failed to connect to local pipeline ctrl port";
        return false;
    }

    if (!runtime_protocol_v2_enabled()) {
        g_pipeline_protocol = DIST_RUNTIME_PROTOCOL_V1;
        runtime_protocol_log_v1_deprecation("node_agent");
        return true;
    }

    uint32_t agreed     = DIST_RUNTIME_PROTOCOL_V1;
    uint32_t server_max = DIST_RUNTIME_PROTOCOL_V1;
    if (!runtime_protocol_negotiate(
                ctrl_fd,
                g_pipeline_session_id,
                DIST_RUNTIME_PROTOCOL_V2,
                agreed,
                server_max)) {
        err = "protocol negotiation failed";
        split_tcp_close(ctrl_fd);
        ctrl_fd = -1;
        return false;
    }

    g_pipeline_protocol = agreed;
    if (agreed < DIST_RUNTIME_PROTOCOL_V2) {
        runtime_protocol_log_v1_deprecation("node_agent");
    }
    if (agreed >= DIST_RUNTIME_PROTOCOL_V2 &&
            !runtime_protocol_exchange_v2_handshake(
                    ctrl_fd, g_pipeline_session_id, agreed, server_max)) {
        err = "protocol v2 handshake failed";
        split_tcp_close(ctrl_fd);
        ctrl_fd = -1;
        return false;
    }
    return true;
}

// Task 19 Phase 3: draft model lives on the node holding `final` and ships
// guesses to entry ahead of time (see split_gen3_a/c.cpp fa-link). The
// client just keeps asking for "the next token" via SPLIT_GEN_CMD_VERIFY
// with a 1-token anchor; entry silently extends the wave from its draft
// buffer when one is ready, so a single request can yield multiple tokens.
static bool speculative_client_enabled() {
    const char * v = std::getenv("DIST_RUNTIME_SPECULATIVE");
    return v != nullptr && v[0] != '\0' && std::strcmp(v, "0") != 0;
}

static bool pipeline_gen3_send_recv(
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

static bool pipeline_gen3_send_hidden_recv(
        int ctrl_fd,
        split_gen_cmd cmd,
        int32_t n_tokens,
        int32_t n_embd,
        int32_t pos_start,
        int32_t layer_end,
        const float * hidden,
        split_gen3_a_resp & resp) {
    if (!split_gen_send_hidden_req(ctrl_fd, cmd, n_tokens, n_embd, pos_start, layer_end, hidden)) {
        return false;
    }
    return split_gen3_recv_a_resp(ctrl_fd, resp, nullptr);
}

static bool run_local_pipeline_generate(
        const std::vector<int32_t> & prompt_tokens,
        int max_new,
        int layer_end,
        std::vector<int32_t> & out_tokens,
        std::string & err,
        json * timing_out = nullptr,
        const std::string * trace_id = nullptr) {
    if (g_pipeline_ctrl_port <= 0) {
        err = "pipeline not configured";
        return false;
    }

    const bool perf_on = perf_trace_enabled() || (trace_id != nullptr && !trace_id->empty());
    struct perf_generate_end_guard {
        bool active = false;
        ~perf_generate_end_guard() {
            if (active) {
                perf_trace_end_generate();
            }
        }
    } generate_guard{};

    struct ttft_end_guard {
        bool ended = false;
        ~ttft_end_guard() {
            if (!ended && perf_trace_enabled()) {
                perf_trace_end_ttft();
            }
        }
    } ttft_guard;

    if (perf_on && trace_id != nullptr && !trace_id->empty()) {
        perf_trace_set_node_id(g_node_id);
        perf_trace_set_component("entry");
        perf_trace_set_context(*trace_id, "ttft", -1);
        perf_trace_refresh_context();
    }

    dist_debug_load_config();
    dist_debug_set_worker("pipeline", g_node_id);
    trace_recorder * dbg = dist_debug_recorder();
    int32_t debug_step   = 0;

    g_runtime_stats.pipeline_generate_count++;
    g_runtime_stats.context_create_count++;

    const auto t_total0 = std::chrono::steady_clock::now();
    int ctrl_fd = -1;
    if (!pipeline_connect_and_negotiate(ctrl_fd, err)) {
        return false;
    }

    split_gen3_a_resp resp{};
    if (dbg) {
        dbg->emit_step_begin(debug_step, "reset", -1, 0, 0);
    }
    if (!pipeline_gen3_send_recv(ctrl_fd, SPLIT_GEN_CMD_RESET, 0, 0, layer_end, nullptr, resp)) {
        err = "reset failed";
        split_tcp_close(ctrl_fd);
        return false;
    }
    g_runtime_stats.kv_cache_reset_count++;

    if (g_external_embedding && !embedding_service_reset(err)) {
        split_tcp_close(ctrl_fd);
        return false;
    }

    const bool use_entry_queue = runtime_entry_queue_client_enabled(g_pipeline_protocol);
    const bool client_pipeline = runtime_client_pipeline_client_enabled(g_pipeline_protocol);

    const auto t_prefill0 = std::chrono::steady_clock::now();
    perf_ttft_span prefill_span("TTFT_PREFILL", "entry");
    if (g_external_embedding) {
        std::vector<float> hidden;
        int32_t n_embd = 0;
        if (!embedding_service_compute(
                    prompt_tokens, 0, hidden, n_embd, err)) {
            split_tcp_close(ctrl_fd);
            return false;
        }
        if (use_entry_queue) {
            if (!pipeline_gen3_roundtrip_hidden_queued(
                        ctrl_fd, SPLIT_GEN_CMD_PREFILL_HIDDEN, (int32_t) prompt_tokens.size(), n_embd,
                        0, layer_end, hidden.data(), resp, false, nullptr)) {
                err = "prefill hidden failed";
                split_tcp_close(ctrl_fd);
                return false;
            }
        } else if (!pipeline_gen3_send_hidden_recv(
                    ctrl_fd, SPLIT_GEN_CMD_PREFILL_HIDDEN, (int32_t) prompt_tokens.size(), n_embd,
                    0, layer_end, hidden.data(), resp)) {
            err = "prefill hidden failed";
            split_tcp_close(ctrl_fd);
            return false;
        }
    } else if (use_entry_queue) {
        if (!pipeline_gen3_roundtrip_queued(
                    ctrl_fd, SPLIT_GEN_CMD_PREFILL, (int32_t) prompt_tokens.size(), 0,
                    layer_end, prompt_tokens.data(), resp, false, nullptr)) {
            err = "prefill failed";
            split_tcp_close(ctrl_fd);
            return false;
        }
    } else if (!pipeline_gen3_send_recv(ctrl_fd, SPLIT_GEN_CMD_PREFILL, (int32_t) prompt_tokens.size(), 0,
            layer_end, prompt_tokens.data(), resp)) {
        err = "prefill failed";
        split_tcp_close(ctrl_fd);
        return false;
    }

    const auto t_first_token = std::chrono::steady_clock::now();

    if (perf_on) {
        char attrs[128];
        std::snprintf(
                attrs,
                sizeof(attrs),
                "{\"token_id\":%d,\"prefill_ms\":%.3f}",
                resp.token_id,
                std::chrono::duration<double, std::milli>(t_first_token - t_prefill0).count());
        perf_emit_ttft_instant("CLIENT_TTFT", "entry", attrs);
        perf_trace_end_ttft();
        ttft_guard.ended = true;
        if (trace_id != nullptr && !trace_id->empty()) {
            perf_trace_begin_generate(*trace_id, "decode");
            generate_guard.active = true;
            perf_trace_set_context(*trace_id, "decode", -1);
            char flag_attrs[224];
            std::snprintf(
                    flag_attrs,
                    sizeof(flag_attrs),
                    "{\"protocol\":%u,\"entry_queue\":%s,\"stage_queue\":%s,"
                    "\"client_pipeline\":%s,\"external_embedding\":%s}",
                    g_pipeline_protocol,
                    use_entry_queue ? "true" : "false",
                    runtime_stage_queue_client_enabled(g_pipeline_protocol) ? "true" : "false",
                    client_pipeline ? "true" : "false",
                    g_external_embedding ? "true" : "false");
            perf_emit_instant("RUNTIME_FLAGS", perf_category::UNKNOWN, "client", -1, flag_attrs);
        }
    }

    if (dbg) {
        dbg->emit_step_begin(debug_step, "prefill", -1, 0, 0);
    }

    if (resp.token_id < 0) {
        err = "prefill returned error from pipeline";
        split_tcp_close(ctrl_fd);
        return false;
    }

    out_tokens.push_back(resp.token_id);
    if (dbg) {
        dbg->emit_token_selected(debug_step, "prefill", resp.token_id,
                (int32_t) prompt_tokens.size() - 1, false);
    }
    debug_step++;
    int32_t cur = resp.token_id;

    const int n_prompt = (int) prompt_tokens.size();
    const bool speculative = speculative_client_enabled() && !use_entry_queue && !g_external_embedding;
    if (speculative) {
        int64_t spec_waves = 0, spec_accepted_total = 0;
        while ((int) out_tokens.size() < max_new) {
            const int32_t pos = n_prompt + (int32_t) out_tokens.size() - 1;
            bool rt_ok;
            {
                perf_span rt_span("CLIENT_BLOCKING_RT_BEGIN", "CLIENT_BLOCKING_RT_END", perf_category::WAIT, "client");
                rt_span.set_token_idx((int32_t) out_tokens.size() - 1);
                rt_ok = pipeline_gen3_send_recv(ctrl_fd, SPLIT_GEN_CMD_VERIFY, 1, pos, layer_end, &cur, resp);
            }
            if (!rt_ok || resp.token_id < 0) {
                err = "verify failed at token " + std::to_string(out_tokens.size());
                split_tcp_close(ctrl_fd);
                return false;
            }
            const int32_t accepted = std::max(0, std::min(resp.accepted_count, SPLIT_GEN_SPEC_MAX_K));
            spec_waves++;
            spec_accepted_total += accepted;
            fprintf(stderr, "SPEC_DEBUG wave=%lld pos=%d accepted=%d\n",
                    (long long) spec_waves, pos, accepted);
            for (int32_t i = 0; i < accepted && (int) out_tokens.size() < max_new; ++i) {
                cur = resp.accepted_ids[i];
                out_tokens.push_back(cur);
                if (dbg) {
                    dbg->emit_token_selected(debug_step, "decode", cur, n_prompt + (int32_t) out_tokens.size() - 2, false);
                }
                debug_step++;
            }
            cur = resp.token_id;
            if ((int) out_tokens.size() < max_new) {
                out_tokens.push_back(cur);
                if (dbg) {
                    dbg->emit_token_selected(debug_step, "decode", cur, n_prompt + (int32_t) out_tokens.size() - 2, false);
                }
                debug_step++;
            }
        }
        split_tcp_close(ctrl_fd);
        fprintf(stderr, "SPEC_DEBUG summary waves=%lld accepted_total=%lld avg_accepted=%.2f\n",
                (long long) spec_waves, (long long) spec_accepted_total,
                spec_waves > 0 ? (double) spec_accepted_total / (double) spec_waves : 0.0);
        const auto t_total1_spec = std::chrono::steady_clock::now();
        if (timing_out) {
            const double prefill_ms = std::chrono::duration<double, std::milli>(t_first_token - t_prefill0).count();
            const double total_ms = std::chrono::duration<double, std::milli>(t_total1_spec - t_total0).count();
            const double decode_ms = std::max(0.0, total_ms - prefill_ms);
            *timing_out = {
                { "prefill_ms", prefill_ms },
                { "ttft_ms", prefill_ms },
                { "decode_ms", decode_ms },
                { "total_ms", total_ms },
                { "generated_tokens", (int) out_tokens.size() },
            };
            (*timing_out)["protocol_version"] = (int) g_pipeline_protocol;
            (*timing_out)["speculative"] = true;
            if (trace_id != nullptr && !trace_id->empty()) {
                (*timing_out)["trace_id"] = *trace_id;
            }
        }
        if (perf_on && generate_guard.active) {
            perf_emit_instant("CLIENT_RESPONSE", perf_category::UNKNOWN, "entry", -1, nullptr);
        }
        return true;
    }

    bool pending_wave = false;
    for (int step = 1; step < max_new; ++step) {
        const int32_t pos = n_prompt + step - 1;
        if (perf_on && trace_id != nullptr && !trace_id->empty()) {
            perf_trace_set_context(*trace_id, "decode", step - 1);
        }
        if (dbg) {
            dbg->emit_step_begin(debug_step, "decode", cur, pos, 0);
        }

        if (!use_entry_queue) {
            if (g_external_embedding) {
                std::vector<float> hidden;
                int32_t n_embd = 0;
                bool embed_ok;
                {
                    perf_span embed_span("CLIENT_EMBED_BEGIN", "CLIENT_EMBED_END", perf_category::COMPUTE, "client");
                    embed_span.set_token_idx(step - 1);
                    embed_ok = embedding_service_compute({ cur }, pos, hidden, n_embd, err);
                }
                if (!embed_ok) {
                    split_tcp_close(ctrl_fd);
                    return false;
                }
                bool rt_ok;
                {
                    perf_span rt_span("CLIENT_BLOCKING_RT_BEGIN", "CLIENT_BLOCKING_RT_END", perf_category::WAIT, "client");
                    rt_span.set_token_idx(step - 1);
                    rt_ok = pipeline_gen3_send_hidden_recv(
                            ctrl_fd, SPLIT_GEN_CMD_DECODE_HIDDEN, 1, n_embd, pos, layer_end,
                            hidden.data(), resp);
                }
                if (!rt_ok) {
                    err = "decode hidden failed at step " + std::to_string(step);
                    split_tcp_close(ctrl_fd);
                    return false;
                }
            } else {
                bool rt_ok;
                {
                    perf_span rt_span("CLIENT_BLOCKING_RT_BEGIN", "CLIENT_BLOCKING_RT_END", perf_category::WAIT, "client");
                    rt_span.set_token_idx(step - 1);
                    rt_ok = pipeline_gen3_send_recv(ctrl_fd, SPLIT_GEN_CMD_DECODE, 1, pos, layer_end, &cur, resp);
                }
                if (!rt_ok) {
                    err = "decode failed at step " + std::to_string(step);
                    split_tcp_close(ctrl_fd);
                    return false;
                }
            }
        } else {
            if (!pending_wave) {
                if (g_external_embedding) {
                    std::vector<float> hidden;
                    int32_t n_embd = 0;
                    bool embed_ok;
                    {
                        perf_span embed_span("CLIENT_EMBED_BEGIN", "CLIENT_EMBED_END", perf_category::COMPUTE, "client");
                        embed_span.set_token_idx(step - 1);
                        embed_ok = embedding_service_compute({ cur }, pos, hidden, n_embd, err);
                    }
                    if (!embed_ok) {
                        split_tcp_close(ctrl_fd);
                        return false;
                    }
                    bool send_ok;
                    {
                        perf_span send_span("CLIENT_SEND_BEGIN", "CLIENT_SEND_END", perf_category::NETWORK, "client");
                        send_span.set_token_idx(step - 1);
                        send_ok = pipeline_send_gen_hidden_req(
                                ctrl_fd, SPLIT_GEN_CMD_DECODE_HIDDEN, 1, n_embd, pos, layer_end,
                                hidden.data());
                    }
                    if (!send_ok) {
                        err = "decode hidden send failed at step " + std::to_string(step);
                        split_tcp_close(ctrl_fd);
                        return false;
                    }
                } else {
                    bool send_ok;
                    {
                        perf_span send_span("CLIENT_SEND_BEGIN", "CLIENT_SEND_END", perf_category::NETWORK, "client");
                        send_span.set_token_idx(step - 1);
                        send_ok = pipeline_send_gen_req(
                                ctrl_fd, SPLIT_GEN_CMD_DECODE, 1, pos, layer_end, &cur);
                    }
                    if (!send_ok) {
                        err = "decode send failed at step " + std::to_string(step);
                        split_tcp_close(ctrl_fd);
                        return false;
                    }
                }
                int32_t depth = 0;
                int32_t wave_id = -1;
                bool ack_ok;
                {
                    perf_span ack_span("CLIENT_ACK_WAIT_BEGIN", "CLIENT_ACK_WAIT_END", perf_category::WAIT, "client");
                    ack_span.set_token_idx(step - 1);
                    ack_ok = pipeline_recv_queue_ack(ctrl_fd, depth, wave_id);
                }
                if (!ack_ok) {
                    err = "decode queue ack failed at step " + std::to_string(step);
                    split_tcp_close(ctrl_fd);
                    return false;
                }
            }

            int32_t token_id = -1;
            int32_t ready_wave = -1;
            bool token_ok;
            {
                perf_span token_span("CLIENT_TOKEN_WAIT_BEGIN", "CLIENT_TOKEN_WAIT_END", perf_category::WAIT, "client");
                token_span.set_token_idx(step - 1);
                token_ok = pipeline_recv_token_ready(ctrl_fd, token_id, ready_wave);
            }
            if (!token_ok) {
                err = "decode token ready failed at step " + std::to_string(step);
                split_tcp_close(ctrl_fd);
                return false;
            }
            cur = token_id;
            resp.token_id = token_id;
            pending_wave = false;

            if (client_pipeline && step + 1 < max_new) {
                const int32_t next_pos = n_prompt + step;
                if (g_external_embedding) {
                    std::vector<float> next_hidden;
                    int32_t next_n_embd = 0;
                    bool embed_ok;
                    {
                        perf_span embed_span("CLIENT_EMBED_BEGIN", "CLIENT_EMBED_END", perf_category::COMPUTE, "client");
                        embed_span.set_token_idx(step);
                        embed_ok = embedding_service_compute({ cur }, next_pos, next_hidden, next_n_embd, err);
                    }
                    if (!embed_ok) {
                        split_tcp_close(ctrl_fd);
                        return false;
                    }
                    bool send_ok;
                    {
                        perf_span send_span("CLIENT_SEND_BEGIN", "CLIENT_SEND_END", perf_category::NETWORK, "client");
                        send_span.set_token_idx(step);
                        send_ok = pipeline_send_gen_hidden_req(
                                ctrl_fd, SPLIT_GEN_CMD_DECODE_HIDDEN, 1, next_n_embd, next_pos,
                                layer_end, next_hidden.data());
                    }
                    if (!send_ok) {
                        err = "decode hidden pipeline send failed at step " + std::to_string(step);
                        split_tcp_close(ctrl_fd);
                        return false;
                    }
                } else {
                    bool send_ok;
                    {
                        perf_span send_span("CLIENT_SEND_BEGIN", "CLIENT_SEND_END", perf_category::NETWORK, "client");
                        send_span.set_token_idx(step);
                        send_ok = pipeline_send_gen_req(
                                ctrl_fd, SPLIT_GEN_CMD_DECODE, 1, next_pos, layer_end, &cur);
                    }
                    if (!send_ok) {
                        err = "decode pipeline send failed at step " + std::to_string(step);
                        split_tcp_close(ctrl_fd);
                        return false;
                    }
                }
                int32_t depth2 = 0;
                int32_t wave2 = -1;
                bool ack_ok;
                {
                    perf_span ack_span("CLIENT_ACK_WAIT_BEGIN", "CLIENT_ACK_WAIT_END", perf_category::WAIT, "client");
                    ack_span.set_token_idx(step);
                    ack_ok = pipeline_recv_queue_ack(ctrl_fd, depth2, wave2);
                }
                if (!ack_ok) {
                    err = "decode pipeline ack failed at step " + std::to_string(step);
                    split_tcp_close(ctrl_fd);
                    return false;
                }
                pending_wave = true;
            }

            if (client_pipeline && step + 1 >= max_new) {
                if (!pipeline_send_drain_pending(ctrl_fd, layer_end)) {
                    err = "decode drain failed at step " + std::to_string(step);
                    split_tcp_close(ctrl_fd);
                    return false;
                }
            }

            bool complete_ok;
            {
                perf_span complete_span("CLIENT_COMPLETE_WAIT_BEGIN", "CLIENT_COMPLETE_WAIT_END", perf_category::WAIT, "client");
                complete_span.set_token_idx(step - 1);
                complete_ok = pipeline_recv_gen_complete(ctrl_fd, resp);
            }
            if (!complete_ok) {
                err = "decode complete failed at step " + std::to_string(step);
                split_tcp_close(ctrl_fd);
                return false;
            }
            if (resp.token_id < 0) {
                resp.token_id = cur;
            }
        }
        if (resp.token_id < 0) {
            err = "pipeline error at step " + std::to_string(step);
            split_tcp_close(ctrl_fd);
            return false;
        }
        cur = resp.token_id;
        out_tokens.push_back(cur);
        if (dbg) {
            dbg->emit_token_selected(debug_step, "decode", cur, pos, false);
        }
        debug_step++;
    }

    split_tcp_close(ctrl_fd);

    const auto t_total1 = std::chrono::steady_clock::now();
    if (timing_out) {
        const double prefill_ms = std::chrono::duration<double, std::milli>(t_first_token - t_prefill0).count();
        const double total_ms = std::chrono::duration<double, std::milli>(t_total1 - t_total0).count();
        const double decode_ms = std::max(0.0, total_ms - prefill_ms);
        *timing_out = {
            { "prefill_ms", prefill_ms },
            { "ttft_ms", prefill_ms },
            { "decode_ms", decode_ms },
            { "total_ms", total_ms },
            { "generated_tokens", (int) out_tokens.size() },
        };
        (*timing_out)["protocol_version"] = (int) g_pipeline_protocol;
        (*timing_out)["entry_queue"] = use_entry_queue;
        (*timing_out)["stage_queue"] = runtime_stage_queue_client_enabled(g_pipeline_protocol);
        (*timing_out)["client_pipeline"] = client_pipeline;
        (*timing_out)["external_embedding"] = g_external_embedding;
        if (trace_id != nullptr && !trace_id->empty()) {
            (*timing_out)["trace_id"] = *trace_id;
        }
        if (const char * dir = perf_trace_output_dir()) {
            (*timing_out)["perf_trace_dir"] = dir;
        }
        if (trace_id != nullptr && !trace_id->empty()) {
            (*timing_out)["ttft_trace_dir"] =
                    perf_trace_config_get().trace_dir + "/" + *trace_id + "/ttft";
        }
    }
    if (perf_on && generate_guard.active) {
        perf_emit_instant("CLIENT_RESPONSE", perf_category::UNKNOWN, "entry", -1, nullptr);
    }
    return true;
}

static bool output_service_ready() {
    std::string err;
    if (!g_output_service_gguf.empty() || !g_output_service_model_id.empty()) {
        return ensure_output_service_loaded(err);
    }
    return !g_output_service_host.empty() && g_output_service_port > 0;
}

static void set_worker_layer_store_env(
        const dist_configure_req & cfg,
        const worker_role role,
        const int32_t layer_start,
        const int32_t layer_end) {
    dist_set_env("DIST_RUNTIME_LAYER_FIRST", "1");
    dist_set_env("DIST_MODEL_ID", cfg.model_id.c_str());
    if (!g_models_dir.empty()) {
        dist_set_env("DIST_LAYER_STORE_ROOT", g_models_dir.c_str());
    }
    dist_set_env("DIST_WORKER_ROLE", worker_role_to_string(role).c_str());
    dist_set_env("DIST_WORKER_LAYER_START", std::to_string(layer_start).c_str());
    dist_set_env("DIST_WORKER_LAYER_END", std::to_string(layer_end).c_str());
}

static bool start_worker(
        const std::string & agent_bin,
        const dist_configure_req & cfg,
        const std::string & model_path,
        std::string & err) {
    stop_worker_for_role(cfg.role);

    if (model_path.empty()) {
        err = "no model path for worker";
        return false;
    }

    int32_t worker_layer_start = cfg.layer_start;
    int32_t worker_layer_end   = cfg.layer_end;
    worker_role worker_load_role = dist_role_to_worker_role(cfg.role);

    if (model_path == LAYER_STORE_MODEL_SENTINEL) {
        if (cfg.model_id.empty()) {
            err = "model_id required for layer-store worker";
            return false;
        }
        set_worker_layer_store_env(cfg, worker_load_role, worker_layer_start, worker_layer_end);
    }

    const std::string worker_arg =
            model_path == LAYER_STORE_MODEL_SENTINEL ? LAYER_STORE_MODEL_SENTINEL : model_path;

    const std::string suffix = dist_exe_suffix();
    const std::string dir    = agent_bin;
    std::string bin;
    std::vector<std::string> args;

    if (cfg.role == DIST_ROLE_ENTRY) {
        bin = dist_join_path(dir, "split_gen3_a" + suffix);
        args = {
            bin, worker_arg,
            "--ctrl-port", std::to_string(cfg.ctrl_port),
            "--b-host", cfg.next_host,
            "--b-port", std::to_string(cfg.next_port),
            "--layer-start", std::to_string(cfg.layer_start),
            "--layer-end", std::to_string(cfg.layer_end),
            "--bind", cfg.peer_bind,
        };
        if (cfg.next_is_final) {
            args.push_back("--next-final");
        }
        if (cfg.fa_port > 0) {
            args.push_back("--fa-port");
            args.push_back(std::to_string(cfg.fa_port));
        }
    } else if (cfg.role == DIST_ROLE_MIDDLE) {
        bin = dist_join_path(dir, "split_gen3_b" + suffix);
        args = {
            bin, worker_arg,
            "--ab-port", std::to_string(cfg.peer_port),
            "--bc-host", cfg.next_host,
            "--bc-port", std::to_string(cfg.next_port),
            "--layer-start", std::to_string(cfg.layer_start),
            "--layer-end", std::to_string(cfg.layer_end),
            "--bind", cfg.peer_bind,
        };
    } else if (cfg.role == DIST_ROLE_FINAL) {
        bin = dist_join_path(dir, "split_gen3_c" + suffix);
        args = {
            bin, worker_arg,
            "--bc-port", std::to_string(cfg.peer_port),
            "--layer-start", std::to_string(cfg.layer_start),
            "--bind", cfg.peer_bind,
        };
        if (cfg.layer_end > cfg.layer_start) {
            args.push_back("--layer-end");
            args.push_back(std::to_string(cfg.layer_end));
        }
        if (g_external_output) {
            args.push_back("--external-output");
            const std::string output_host =
                    !cfg.output_service_host.empty() ? cfg.output_service_host : g_output_service_host;
            const int output_port =
                    cfg.output_service_port > 0 ? cfg.output_service_port : g_output_service_port;
            if (!output_host.empty()) {
                args.push_back("--output-http-host");
                args.push_back(output_host);
            }
            if (output_port > 0) {
                args.push_back("--output-http-port");
                args.push_back(std::to_string(output_port));
            }
        }
        if (!cfg.draft_model.empty() && cfg.fa_port > 0) {
            args.push_back("--draft-model");
            args.push_back(cfg.draft_model);
            args.push_back("--fa-host");
            args.push_back(cfg.fa_host);
            args.push_back("--fa-port");
            args.push_back(std::to_string(cfg.fa_port));
            args.push_back("--draft-k");
            args.push_back(std::to_string(cfg.draft_k));
        }
    } else {
        err = "invalid role";
        return false;
    }

    if (dist_debug_enabled()) {
        dist_set_env("LLAMA_DISTRIBUTED_DEBUG", "1");
        if (!cfg.session_id.empty()) {
            dist_set_env("LLAMA_DIST_SESSION_ID", cfg.session_id.c_str());
        }
        dist_set_env("LLAMA_DIST_WORKER_ID", dist_role_name(cfg.role).c_str());
        dist_set_env("LLAMA_DIST_NODE_ID", g_node_id.c_str());
        if (!g_models_dir.empty()) {
            const std::string trace_dir = g_models_dir + "/traces";
            dist_set_env("LLAMA_DIST_TRACE_DIR", trace_dir.c_str());
        }
    }
    // Always give workers their perf identity: tracing may be enabled after
    // spawn (begin_decode), and events without node_id/component are
    // unattributable in analysis.
    dist_set_env("DIST_NODE_ID", g_node_id.c_str());
    dist_set_env("DIST_PERF_COMPONENT", dist_role_name(cfg.role).c_str());
    if (!g_models_dir.empty()) {
        const std::string perf_dir = g_models_dir + "/perf_trace";
        dist_set_env("DIST_PERF_TRACE_DIR", perf_dir.c_str());
    }
    if (perf_trace_enabled()) {
        dist_set_env("DIST_PERF_TRACE", "1");
    }

    const std::string ready_file = worker_ready_state_file(cfg.role, cfg.session_id);
    std::remove(ready_file.c_str());
    args.push_back("--ready-file");
    args.push_back(ready_file);

    std::string worker_log_path;
    if (!g_models_dir.empty()) {
        std::error_code ec;
        std::filesystem::create_directories(g_models_dir + "/logs", ec);
        worker_log_path = g_models_dir + "/logs/worker_" + dist_role_name(cfg.role) + ".log";
    }

    dist_child_process child{};
    if (!dist_process_spawn(args, child, err, worker_log_path)) {
        return false;
    }

    if (dist_child_process * slot = worker_proc_slot(cfg.role)) {
        *slot = child;
    }
    if (std::string * state_slot = worker_state_file_slot(cfg.role)) {
        *state_slot = ready_file;
    }
    g_runtime_stats.worker_spawn_count++;
    std::string final_state;
    if (!wait_worker_ready_state(cfg.role, ready_file, "LISTENER_READY", 120000, final_state, err)) {
        stop_worker_for_role(cfg.role);
        return false;
    }
    return true;
}

static worker_role dist_role_to_worker_role(const dist_node_role role) {
    switch (role) {
        case DIST_ROLE_ENTRY:
        case DIST_ROLE_MIDDLE:
        case DIST_ROLE_FINAL:
            return worker_role::pipeline_stage;
        default:
            return worker_role::pipeline_stage;
    }
}

static bool tokenizer_shell_loadable(const std::string & path) {
    if (path.empty()) {
        return false;
    }
    ggml_backend_load_all();
    llama_model * model = llama_model_load_from_file(path.c_str(), default_tokenizer_model_params());
    if (!model) {
        return false;
    }
    llama_model_free(model);
    return true;
}

static bool embedding_shell_loadable(const std::string & path) {
    if (path.empty()) {
        return false;
    }
    ggml_backend_load_all();
    llama_model * model = llama_model_load_from_file(path.c_str(), llama_model_default_params());
    if (!model) {
        return false;
    }
    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx   = 8;
    cparams.n_batch = 8;
    cparams.no_perf = true;
    llama_context * ctx = llama_init_from_model(model, cparams);
    if (!ctx) {
        llama_model_free(model);
        return false;
    }
    llama_set_layer_range(ctx, 0, 1);
    llama_batch batch = llama_batch_init(1, 0, 1);
    batch.token[0]     = 1;
    batch.pos[0]       = 0;
    batch.n_seq_id[0]  = 1;
    batch.seq_id[0][0] = 0;
    batch.logits[0]    = 1;
    batch.n_tokens     = 1;
    const bool ok = llama_decode(ctx, batch) == 0 && llama_get_embeddings(ctx) != nullptr;
    llama_batch_free(batch);
    llama_free(ctx);
    llama_model_free(model);
    return ok;
}

static bool output_shell_loadable(const std::string & path) {
    if (path.empty()) {
        return false;
    }
    ggml_backend_load_all();
    llama_model * model = llama_model_load_from_file(path.c_str(), llama_model_default_params());
    if (!model) {
        return false;
    }
    const int32_t n_layer = llama_model_n_layer(model);
    const int32_t n_embd  = llama_model_n_embd(model);
    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx   = 8;
    cparams.n_batch = 8;
    cparams.no_perf = true;
    llama_context * ctx = llama_init_from_model(model, cparams);
    if (!ctx) {
        llama_model_free(model);
        return false;
    }
    llama_set_layer_range(ctx, n_layer - 1, n_layer);
    std::vector<float> dummy((size_t) n_embd, 0.0f);
    const bool ok = split_gen_decode_hidden(ctx, dummy.data(), 1, n_embd, 0, true, true) == 0 &&
            llama_get_logits(ctx) != nullptr;
    llama_free(ctx);
    llama_model_free(model);
    return ok;
}

static std::string configure_worker_runtime(
        const dist_configure_req & cfg,
        const json & body,
        std::string & err,
        runtime_worker_gguf_resolve_result * resolve_out = nullptr) {
    runtime_worker_gguf_request req{};
    req.cfg = cfg;
    req.body = body;
    req.verify_materialization = g_verify_materialization;
    req.cache_shell_ok = [](const std::string & path, const worker_role role) -> bool {
        if (role == worker_role::tokenizer) {
            return tokenizer_shell_loadable(path);
        }
        if (role == worker_role::embedding) {
            return embedding_shell_loadable(path);
        }
        if (role == worker_role::output_head) {
            return output_shell_loadable(path);
        }
        return true;
    };

    const auto resolved = runtime_resolve_worker_gguf(get_layer_store(cfg.model_id), req, err);
    if (resolve_out != nullptr) {
        *resolve_out = resolved;
    }
    if (!err.empty()) {
        return {};
    }

    if (resolved.materialize_role == worker_role::sampler) {
        g_sampler_service_ready = true;
        return {};
    }

    if (resolved.worker_gguf_path.empty()) {
        if (resolved.bind.tensors_ready) {
            err.clear();
            return {};
        }
        err = "runtime resolve produced no worker artifact";
        return {};
    }

    if (resolved.compat_materialized) {
        if (resolved.materialize_role == worker_role::tokenizer &&
                !tokenizer_shell_loadable(resolved.worker_gguf_path)) {
            err = "tokenizer gguf not loadable after materialize";
            return {};
        }
        if (resolved.materialize_role == worker_role::embedding &&
                !embedding_shell_loadable(resolved.worker_gguf_path)) {
            err = "embedding gguf not loadable after materialize";
            return {};
        }
        if (resolved.materialize_role == worker_role::output_head &&
                !output_shell_loadable(resolved.worker_gguf_path)) {
            err = "output gguf not loadable after materialize";
            return {};
        }
        g_runtime_stats.materialization_count++;
        g_runtime_stats.materialization_generation++;
    }

    return resolved.worker_gguf_path;
}

int main(int argc, char ** argv) {
    std::string listen;
    std::string orchestrator;
    std::string advertise_host;
    bool rebenchmark = false;

    if (!parse_args(argc, argv, listen, orchestrator, g_node_id, advertise_host, rebenchmark)) {
        usage(argv[0]);
        return 1;
    }

    std::string bind_host;
    int http_port = 0;
    if (!dist_parse_host_port(listen, bind_host, http_port)) {
        fprintf(stderr, "node_agent: invalid --listen %s\n", listen.c_str());
        return 1;
    }
    g_agent_http_port = http_port;

    if (g_node_id.empty()) {
        g_node_id = bind_host + ":" + std::to_string(http_port);
    }

    // Layer-first is this project's default runtime mode (Task 11); GGUF
    // materialization is a compat fallback. runtime_layer_first_enabled()
    // reads DIST_RUNTIME_LAYER_FIRST from THIS process's own environment,
    // but the only place that ever set it was set_worker_layer_store_env()
    // inside the /configure handler -- which runs strictly after
    // /runtime/prepare, the handler that actually reads the flag to decide
    // materialize-vs-bind. On a freshly started node_agent, every worker's
    // very first /runtime/prepare therefore always saw the flag unset and
    // fell through to materializing a full worker GGUF via the (separately
    // buggy) legacy assembler. Set the default here, at process start,
    // before any request can be handled, so the decision point sees it.
    // An operator can still force legacy materialization with
    // DIST_RUNTIME_LAYER_FIRST=0 in the environment before launch.
    if (std::getenv("DIST_RUNTIME_LAYER_FIRST") == nullptr) {
        dist_set_env("DIST_RUNTIME_LAYER_FIRST", "1");
    }

    dist_debug_load_config();
    dist_debug_set_worker("node_agent", g_node_id);
    split_tcp_init();

    if (!g_model_path.empty()) {
        fprintf(stderr, "node_agent: running benchmark on %s\n", g_model_path.c_str());
    } else {
        fprintf(stderr, "node_agent: no local model — hardware-only benchmark\n");
    }
    g_benchmark = dist_get_or_run_benchmark_optional(g_model_path, rebenchmark);
    if (g_benchmark.score <= 0.0f) {
        fprintf(stderr, "node_agent: benchmark failed\n");
        return 1;
    }

    g_n_layer = g_benchmark.n_layer;
    g_n_embd  = g_benchmark.n_embd;
    if (g_n_layer <= 0 || g_n_embd <= 0) {
        if (!g_model_path.empty() && !load_model_metadata()) {
            fprintf(stderr, "node_agent: failed to read model metadata from %s\n", g_model_path.c_str());
            return 1;
        }
    }

    // Initialize model store
    if (!g_models_dir.empty()) {
        g_model_store.set_models_dir(g_models_dir);
    }
    g_models_state_path = g_model_store.get_models_dir() + "/models.json";
    if (!g_model_store.load_state(g_models_state_path)) {
        fprintf(stderr, "node_agent: warning - failed to load model state from %s\n", g_models_state_path.c_str());
    }

    std::string register_host = !advertise_host.empty() ? advertise_host : bind_host;
    g_register_host = register_host;
    if (register_host == "0.0.0.0" || register_host == "*") {
        fprintf(stderr, "node_agent: set --advertise-host to a reachable IP (not 0.0.0.0)\n");
        return 1;
    }

    // The orchestrator and this node are commonly restarted together (e.g.
    // after a rebuild); a bare orchestrator start lag or brief network blip
    // at boot must not be fatal here -- only the ongoing heartbeat below
    // retries on its own, and it never runs if this first call gives up.
    {
        const int max_attempts = 15;
        const int retry_delay_s = 2;
        bool registered = false;
        for (int attempt = 1; attempt <= max_attempts; ++attempt) {
            if (register_with_orchestrator(orchestrator, g_node_id, register_host, http_port)) {
                registered = true;
                break;
            }
            if (attempt < max_attempts) {
                fprintf(stderr, "node_agent: register attempt %d/%d failed, retrying in %ds\n",
                        attempt, max_attempts, retry_delay_s);
                std::this_thread::sleep_for(std::chrono::seconds(retry_delay_s));
            }
        }
        if (!registered) {
            fprintf(stderr, "node_agent: giving up after %d register attempts\n", max_attempts);
            return 1;
        }
    }

    // Heartbeat: periodically re-register so the orchestrator recovers this node
    // after an orchestrator restart (registration is upserted by node_id).
    std::thread([orchestrator, register_host, http_port]() {
        for (;;) {
            std::this_thread::sleep_for(std::chrono::seconds(5));
            register_with_orchestrator(orchestrator, g_node_id, register_host, http_port);
        }
    }).detach();

    // Network observability (Task 19 level 2): ping peers' /health once a
    // second and track RTT/loss per peer, refreshing the peer list from
    // the orchestrator every ~5s. See peer_rtt_tracker above.
    std::thread([orchestrator]() {
        std::map<std::string, std::pair<std::string, int>> peers;
        int refresh_in = 0;
        for (;;) {
            std::this_thread::sleep_for(std::chrono::seconds(1));
            if (refresh_in <= 0) {
                httplib::Client cli(orchestrator.c_str());
                cli.set_connection_timeout(3, 0);
                cli.set_read_timeout(3, 0);
                if (auto res = cli.Get("/nodes")) {
                    if (res->status == 200) {
                        try {
                            const json d = json::parse(res->body);
                            std::map<std::string, std::pair<std::string, int>> next_peers;
                            for (const auto & n : d.value("nodes", json::array())) {
                                const std::string nid = n.value("node_id", "");
                                if (nid.empty() || nid == g_node_id) {
                                    continue;
                                }
                                next_peers[nid] = { n.value("host", ""), n.value("port", 0) };
                            }
                            peers = std::move(next_peers);
                        } catch (...) {
                            // keep the previous peer list on a parse error
                        }
                    }
                }
                refresh_in = 5;
            }
            --refresh_in;

            for (const auto & kv : peers) {
                const std::string & nid  = kv.first;
                const std::string & host = kv.second.first;
                const int           port = kv.second.second;
                if (host.empty() || port <= 0) {
                    continue;
                }
                httplib::Client cli(host.c_str(), port);
                cli.set_connection_timeout(0, 500000);
                cli.set_read_timeout(0, 500000);
                const auto t0 = std::chrono::steady_clock::now();
                auto res = cli.Get("/health");
                std::lock_guard<std::mutex> lock(g_peer_rtt_mu);
                auto & tracker = g_peer_rtt[nid];
                if (!tracker) {
                    tracker = std::make_unique<peer_rtt_tracker>();
                }
                if (res && res->status == 200) {
                    const double rtt_ms = std::chrono::duration<double, std::milli>(
                            std::chrono::steady_clock::now() - t0).count();
                    tracker->add_success(rtt_ms);
                } else {
                    tracker->add_failure();
                }
            }
        }
    }).detach();

    const std::string agent_bin = exe_dir(argv[0]);

    httplib::Server svr;

    // CORS: the dashboard app (Electron renderer) fetches other nodes'
    // node_agent directly from the browser, a different origin than this
    // server -- without these headers the browser blocks the response.
    svr.set_post_routing_handler([](const httplib::Request &, httplib::Response & res) {
        res.set_header("Access-Control-Allow-Origin", "*");
        res.set_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS");
        res.set_header("Access-Control-Allow-Headers", "Content-Type");
    });
    svr.Options(R"(.*)", [](const httplib::Request &, httplib::Response & res) {
        res.status = 204;
    });

    svr.Get("/health", [](const httplib::Request &, httplib::Response & res) {
        res.set_content(json({ { "status", "ok" }, { "node_id", g_node_id } }).dump(), "application/json");
    });

    svr.Get("/network/stats", [](const httplib::Request &, httplib::Response & res) {
        json peers = json::object();
        std::lock_guard<std::mutex> lock(g_peer_rtt_mu);
        for (auto & kv : g_peer_rtt) {
            peers[kv.first] = kv.second->stats();
        }
        res.set_content(json({ { "node_id", g_node_id }, { "peers", peers } }).dump(), "application/json");
    });

    svr.Get("/perf/trace/list", [](const httplib::Request &, httplib::Response & res) {
        const std::string root = perf_trace_resolve_root(g_models_dir);
        res.set_content(json({
            { "node_id", g_node_id },
            { "root", root },
            { "files", perf_trace_list_files(root) },
        }).dump(), "application/json");
    });

    svr.Get("/perf/trace/file", [](const httplib::Request & req, httplib::Response & res) {
        const std::string rel = req.get_param_value("rel");
        const std::string root = perf_trace_resolve_root(g_models_dir);
        std::string content;
        if (!perf_trace_read_file(root, rel, content)) {
            res.status = 404;
            res.set_content(R"({"error":"not found"})", "application/json");
            return;
        }
        res.set_header("Content-Type", "application/x-ndjson");
        res.set_content(content, "application/x-ndjson");
    });

    svr.Post("/perf/trace/cleanup", [](const httplib::Request & req, httplib::Response & res) {
        int max_age_days = 7;
        try {
            if (!req.body.empty()) {
                const json body = json::parse(req.body);
                max_age_days = body.value("max_age_days", max_age_days);
            }
        } catch (...) {
            // Fall through with the default.
        }
        const std::string root = perf_trace_resolve_root(g_models_dir);
        const perf_trace_cleanup_result r = perf_trace_cleanup(root, max_age_days);
        res.set_content(json({
            { "ok", true },
            { "node_id", g_node_id },
            { "root", root },
            { "max_age_days", max_age_days },
            { "deleted_files", r.deleted_files },
            { "freed_bytes", r.freed_bytes },
        }).dump(), "application/json");
    });

    svr.Get("/debug/log", [](const httplib::Request & req, httplib::Response & res) {
        std::string log_file = "node_agent.log";
        if (const std::string w = req.get_param_value("worker"); !w.empty()) {
            static const std::vector<std::string> valid_roles = { "entry", "middle", "final" };
            if (std::find(valid_roles.begin(), valid_roles.end(), w) != valid_roles.end()) {
                log_file = "worker_" + w + ".log";
            }
        }
        const std::string path = log_file == "node_agent.log"
                ? node_log_resolve_path(g_models_dir, log_file)
                : g_models_dir + "/logs/" + log_file;
        if (path.empty() || !std::filesystem::exists(path)) {
            res.status = 404;
            res.set_content(json({ { "error", "log file not found" }, { "path", path } }).dump(),
                    "application/json");
            return;
        }
        size_t lines = 300;
        if (const std::string v = req.get_param_value("lines"); !v.empty()) {
            lines = (size_t) std::max(0, std::atoi(v.c_str()));
        }
        res.set_header("Content-Type", "text/plain");
        res.set_content(node_log_tail(path, lines, 4 * 1024 * 1024), "text/plain");
    });

    svr.Post("/perf/trace/begin_decode", [](const httplib::Request & req, httplib::Response & res) {
        json body;
        try {
            body = json::parse(req.body);
        } catch (...) {
            res.status = 400;
            res.set_content(R"({"ok":false,"error":"invalid json"})", "application/json");
            return;
        }
        const std::string trace_id = body.value("trace_id", "");
        if (trace_id.empty()) {
            res.status = 400;
            res.set_content(R"({"ok":false,"error":"trace_id required"})", "application/json");
            return;
        }
        if (body.value("perf_trace", false)) {
            dist_set_env("DIST_PERF_TRACE", "1");
            if (!g_models_dir.empty()) {
                dist_set_env("DIST_PERF_TRACE_DIR", (g_models_dir + "/perf_trace").c_str());
            }
            // The config is cached at first use; without a reload this process
            // keeps enabled=false and worker spawns skip trace env propagation.
            perf_trace_reload_config();
        }
        perf_trace_set_node_id(g_node_id);
        perf_trace_write_decode_context(trace_id);
        perf_trace_set_context(trace_id, "decode", -1);
        res.set_content(json({
            { "ok", true },
            { "trace_id", trace_id },
            { "node_id", g_node_id },
        }).dump(), "application/json");
    });

    svr.Get("/capabilities", [](const httplib::Request &, httplib::Response & res) {
        dist_node_memory memory{};
        dist_node_cpu cpu{};
        const dist_node_system sys = dist_probe_node_system();
        dist_probe_node_memory(memory);
        dist_probe_node_cpu(cpu);
        const auto caps = dist_probe_capabilities();

        res.set_content(json({
            { "node_id", g_node_id },
            { "memory", {
                { "total_ram", memory.total_ram_bytes },
                { "free_ram", memory.free_ram_bytes },
                { "total_vram", memory.total_vram_bytes },
                { "free_vram", memory.free_vram_bytes },
                { "has_gpu", memory.has_gpu },
            }},
            { "hardware", {
                { "backend", caps.gpu_backend },
                { "gpu_name", caps.gpu_name },
                { "cpu_name", cpu.cpu_name },
                { "logical_cores", cpu.logical_cores },
                { "physical_cores", cpu.physical_cores },
            }},
            { "system", {
                { "os", sys.os },
                { "arch", sys.arch },
            }},
            { "performance", {
                { "score", g_benchmark.score },
                { "decode_tps", g_benchmark.decode_tps },
                { "prefill_tps", g_benchmark.prefill_tps },
                { "load_ms", g_benchmark.load_ms },
            }},
        }).dump(), "application/json");
    });

    svr.Get("/status", [](const httplib::Request &, httplib::Response & res) {
        res.set_content(json({
            { "node_id", g_node_id },
            { "worker_pid", (int) g_entry_worker.pid },
            { "workers", {
                { "entry", (int) g_entry_worker.pid },
                { "middle", (int) g_middle_worker.pid },
                { "final", (int) g_final_worker.pid },
            }},
            { "worker_states", worker_ready_states_json() },
            { "runtime_stats", node_runtime_stats_json() },
        }).dump(), "application/json");
    });

    svr.Post("/runtime/tokenizer/configure", [](const httplib::Request & req, httplib::Response & res) {
        json body;
        try {
            body = json::parse(req.body);
        } catch (...) {
            res.status = 400;
            res.set_content(R"({"error":"invalid json"})", "application/json");
            return;
        }
        perf_consume_trace_from_body(body, "tokenizer");
        perf_session_span svc_span("SESSION_SERVICE_CONFIGURE", g_node_id.c_str(), "tokenizer");
        const std::string path = body.value("worker_gguf", "");
        const std::string model_id = body.value("model_id", "");
        if (!apply_service_model_source(path, model_id, g_tokenizer_service_gguf, g_tokenizer_service_model_id)) {
            res.status = 400;
            res.set_content(R"({"ok":false,"error":"worker_gguf or model_id required"})", "application/json");
            return;
        }
        free_tokenizer_service();
        const bool ready = tokenizer_service_vocab() != nullptr;
        res.set_content(json({
            { "ok", ready },
            { "tokenizer_ready", ready },
        }).dump(), "application/json");
    });

    svr.Post("/runtime/tokenizer/tokenize", [](const httplib::Request & req, httplib::Response & res) {
        json body;
        try {
            body = json::parse(req.body);
        } catch (...) {
            res.status = 400;
            res.set_content(R"({"error":"invalid json"})", "application/json");
            return;
        }
        const std::string prompt = body.value("prompt", "");
        const llama_vocab * vocab = tokenizer_service_vocab();
        if (!vocab) {
            res.status = 503;
            res.set_content(R"({"ok":false,"error":"tokenizer service not configured"})", "application/json");
            return;
        }
        const std::vector<llama_token> toks = split_gen_tokenize(vocab, prompt);
        json tokens = json::array();
        for (const llama_token t : toks) {
            tokens.push_back((int32_t) t);
        }
        res.set_content(json({ { "ok", true }, { "tokens", tokens } }).dump(), "application/json");
    });

    svr.Post("/runtime/tokenizer/detokenize", [](const httplib::Request & req, httplib::Response & res) {
        json body;
        try {
            body = json::parse(req.body);
        } catch (...) {
            res.status = 400;
            res.set_content(R"({"error":"invalid json"})", "application/json");
            return;
        }
        const llama_vocab * vocab = tokenizer_service_vocab();
        if (!vocab || !body.contains("tokens") || !body["tokens"].is_array()) {
            res.status = 400;
            res.set_content(R"({"ok":false,"error":"tokenizer or tokens missing"})", "application/json");
            return;
        }
        std::string text;
        for (const auto & t : body["tokens"]) {
            text += split_gen_token_text(vocab, (llama_token) t.get<int32_t>());
        }
        res.set_content(json({ { "ok", true }, { "text", text } }).dump(), "application/json");
    });

    svr.Post("/runtime/bind", [](const httplib::Request & req, httplib::Response & res) {
        json body;
        try {
            body = json::parse(req.body);
        } catch (...) {
            res.status = 400;
            res.set_content(R"({"error":"invalid json"})", "application/json");
            return;
        }
        dist_configure_req cfg = parse_configure(body);
        if (cfg.model_id.empty()) {
            res.status = 400;
            res.set_content(R"({"ok":false,"error":"model_id required"})", "application/json");
            return;
        }
        layer_store store = get_layer_store(cfg.model_id);
        const auto manifest = store.load_manifest();
        if (!manifest.has_value()) {
            res.status = 404;
            res.set_content(R"({"ok":false,"error":"manifest not found"})", "application/json");
            return;
        }
        worker_role role = dist_role_to_worker_role(cfg.role);
        if (body.contains("runtime_role")) {
            const runtime_role rt = runtime_role_from_string(body.value("runtime_role", ""));
            if (rt != runtime_role::unassigned) {
                role = worker_role_from_runtime_role(rt);
            }
        }
        const runtime_bind_result bind = runtime_bind_worker(
                store, *manifest, role, cfg.layer_start, cfg.layer_end);
        res.set_content(json({
            { "ok", bind.success || bind.tensors_ready },
            { "tensors_ready", bind.tensors_ready },
            { "cached_gguf_ready", bind.cached_gguf_ready },
            { "materialize_required", bind.materialize_required },
            { "worker_gguf", bind.worker_gguf_path },
            { "error", bind.error },
        }).dump(), "application/json");
    });

    svr.Post("/runtime/embedding/configure", [](const httplib::Request & req, httplib::Response & res) {
        json body;
        try {
            body = json::parse(req.body);
        } catch (...) {
            res.status = 400;
            res.set_content(R"({"error":"invalid json"})", "application/json");
            return;
        }
        perf_consume_trace_from_body(body, "embedding");
        perf_session_span svc_span("SESSION_SERVICE_CONFIGURE", g_node_id.c_str(), "embedding");
        const std::string path = body.value("worker_gguf", "");
        const std::string model_id = body.value("model_id", "");
        if (!apply_service_model_source(path, model_id, g_embedding_service_gguf, g_embedding_service_model_id)) {
            res.status = 400;
            res.set_content(R"({"ok":false,"error":"worker_gguf or model_id required"})", "application/json");
            return;
        }
        g_embedding_service_layer_start = 0;
        g_embedding_service_layer_end   = 0;
        free_embedding_service();
        std::string err;
        const bool ready = ensure_embedding_service_loaded(err);
        if (!ready) {
            res.status = 500;
            res.set_content(json({
                { "ok", false },
                { "embedding_ready", false },
                { "error", err.empty() ? "embedding shell failed to load" : err },
            }).dump(), "application/json");
            return;
        }
        res.set_content(json({
            { "ok", true },
            { "embedding_ready", true },
            { "n_embd", g_embedding_service_n_embd },
        }).dump(), "application/json");
    });

    svr.Post("/runtime/embedding/reset", [](const httplib::Request &, httplib::Response & res) {
        std::string err;
        if (!embedding_service_reset(err)) {
            res.status = 503;
            res.set_content(json({ { "ok", false }, { "error", err } }).dump(), "application/json");
            return;
        }
        res.set_content(R"({"ok":true})", "application/json");
    });

    svr.Post("/runtime/embedding/embed", [](const httplib::Request & req, httplib::Response & res) {
        json body;
        try {
            body = json::parse(req.body);
        } catch (...) {
            res.status = 400;
            res.set_content(R"({"error":"invalid json"})", "application/json");
            return;
        }
        std::vector<int32_t> tokens;
        if (body.contains("tokens") && body["tokens"].is_array()) {
            for (const auto & t : body["tokens"]) {
                tokens.push_back(t.get<int32_t>());
            }
        }
        if (tokens.empty()) {
            res.status = 400;
            res.set_content(R"({"ok":false,"error":"tokens required"})", "application/json");
            return;
        }
        const int32_t pos_start = body.value("pos_start", 0);
        std::vector<float> hidden;
        int32_t n_embd = 0;
        std::string err;
        if (!embedding_service_compute_local(tokens, pos_start, hidden, n_embd, err)) {
            res.status = 503;
            res.set_content(json({ { "ok", false }, { "error", err } }).dump(), "application/json");
            return;
        }
        json hidden_json = json::array();
        for (float v : hidden) {
            hidden_json.push_back(v);
        }
        res.set_content(json({
            { "ok", true },
            { "hidden", hidden_json },
            { "n_embd", n_embd },
            { "n_tokens", (int) tokens.size() },
        }).dump(), "application/json");
    });

    svr.Post("/runtime/output/configure", [](const httplib::Request & req, httplib::Response & res) {
        json body;
        try {
            body = json::parse(req.body);
        } catch (...) {
            res.status = 400;
            res.set_content(R"({"error":"invalid json"})", "application/json");
            return;
        }
        perf_consume_trace_from_body(body, "output");
        perf_session_span svc_span("SESSION_SERVICE_CONFIGURE", g_node_id.c_str(), "output_head");
        const std::string path = body.value("worker_gguf", "");
        const std::string model_id = body.value("model_id", "");
        if (!apply_service_model_source(path, model_id, g_output_service_gguf, g_output_service_model_id)) {
            res.status = 400;
            res.set_content(R"({"ok":false,"error":"worker_gguf or model_id required"})", "application/json");
            return;
        }
        g_output_service_layer_start = 0;
        g_output_service_layer_end   = 0;
        free_output_service();
        std::string err;
        const bool ready = ensure_output_service_loaded(err);
        if (!ready) {
            res.status = 500;
            res.set_content(json({
                { "ok", false },
                { "output_ready", false },
                { "error", err.empty() ? "output shell failed to load" : err },
            }).dump(), "application/json");
            return;
        }
        res.set_content(json({
            { "ok", true },
            { "output_ready", true },
            { "n_vocab", g_output_service_n_vocab },
        }).dump(), "application/json");
    });

    svr.Post("/runtime/output/reset", [](const httplib::Request &, httplib::Response & res) {
        output_service_reset_local();
        res.set_content(R"({"ok":true})", "application/json");
    });

    svr.Post("/runtime/output/sample", [](const httplib::Request & req, httplib::Response & res) {
        json body;
        try {
            body = json::parse(req.body);
        } catch (...) {
            res.status = 400;
            res.set_content(R"({"ok":false,"error":"invalid json"})", "application/json");
            return;
        }
        std::vector<float> hidden;
        if (body.contains("hidden") && body["hidden"].is_array()) {
            for (const auto & v : body["hidden"]) {
                hidden.push_back(v.get<float>());
            }
        }
        const int32_t n_embd = body.value("n_embd", (int32_t) hidden.size());
        const int32_t pos    = body.value("pos", 0);
        if (hidden.empty() || n_embd <= 0) {
            res.status = 400;
            res.set_content(R"({"ok":false,"error":"hidden required"})", "application/json");
            return;
        }
        int32_t token_id = -1;
        std::string err;
        if (!output_service_sample_local(hidden.data(), n_embd, pos, token_id, err)) {
            res.status = 503;
            res.set_content(json({ { "ok", false }, { "error", err } }).dump(), "application/json");
            return;
        }
        res.set_content(json({
            { "ok", true },
            { "token_id", token_id },
            { "n_vocab", g_output_service_n_vocab },
        }).dump(), "application/json");
    });

    svr.Post("/runtime/sampler/configure", [](const httplib::Request & req, httplib::Response & res) {
        json body;
        try {
            body = json::parse(req.body);
        } catch (...) {
            body = json::object();
        }
        perf_consume_trace_from_body(body, "sampler");
        perf_session_span svc_span("SESSION_SERVICE_CONFIGURE", g_node_id.c_str(), "sampler");
        g_sampler_service_ready = true;
        res.set_content(json({ { "ok", true }, { "sampler_ready", true } }).dump(), "application/json");
    });

    // Task 19 Phase 3: fetch the speculative-decoding draft model. Unlike
    // /runtime/prepare (registry + manifest + layer-store, built for
    // splitting one model's layers across nodes), the draft is a single
    // small whole model with a URL the caller already knows -- no
    // registration or slicing needed, just get the file onto this node.
    svr.Post("/draft/fetch", [](const httplib::Request & req, httplib::Response & res) {
        json body;
        try {
            body = json::parse(req.body);
        } catch (...) {
            res.status = 400;
            res.set_content(R"({"error":"invalid json"})", "application/json");
            return;
        }
        const std::string url      = body.value("source_url", "");
        const std::string filename = body.value("filename", "");
        if (url.empty() || filename.empty()) {
            res.status = 400;
            res.set_content(R"({"error":"source_url and filename required"})", "application/json");
            return;
        }

        const std::string dest_dir = g_models_dir.empty() ? "/tmp" : g_models_dir + "/draft";
        std::error_code ec;
        std::filesystem::create_directories(dest_dir, ec);
        const std::string dest_path = dest_dir + "/" + filename;

        if (!std::filesystem::exists(dest_path, ec) || std::filesystem::file_size(dest_path, ec) == 0) {
            std::string derr;
            if (!dist_http_download_file(url, dest_path, derr)) {
                res.status = 500;
                res.set_content(json({ { "ok", false }, { "error", derr } }).dump(), "application/json");
                return;
            }
        }
        res.set_content(json({ { "ok", true }, { "path", dest_path } }).dump(), "application/json");
    });

    svr.Post("/runtime/prepare", [](const httplib::Request & req, httplib::Response & res) {
        json body;
        try {
            body = json::parse(req.body);
        } catch (...) {
            res.status = 400;
            res.set_content(R"({"error":"invalid json"})", "application/json");
            return;
        }

        perf_consume_trace_from_body(body, "prepare");
        const std::string rt_role = body.value("runtime_role", "");
        const std::string role = body.value("role", rt_role.empty() ? "pipeline" : rt_role);
        perf_session_span prep_span("SESSION_PREPARE_RUNTIME", g_node_id.c_str(), role.c_str());

        const auto t0 = std::chrono::steady_clock::now();
        dist_configure_req cfg = parse_configure(body);
        std::string err;
        runtime_worker_gguf_resolve_result resolved{};
        const std::string worker_gguf =
                configure_worker_runtime(cfg, body, err, &resolved);
        const bool is_sampler = rt_role == "sampler" ||
                resolved.materialize_role == worker_role::sampler;
        const bool bind_ready = worker_gguf.empty() && err.empty() && resolved.bind.tensors_ready;
        if (worker_gguf.empty() && !err.empty()) {
            res.status = 500;
            res.set_content(json({
                { "status", "error" },
                { "runtime_ready", false },
                { "error", err },
            }).dump(), "application/json");
            return;
        }
        if (worker_gguf.empty() && !is_sampler && !bind_ready) {
            res.status = 500;
            res.set_content(json({
                { "status", "error" },
                { "runtime_ready", false },
                { "error", "prepare produced no worker artifact" },
            }).dump(), "application/json");
            return;
        }

        bool tokenizer_ready = false;
        if (rt_role == "tokenizer") {
            if (worker_gguf.empty() && bind_ready) {
                g_tokenizer_service_gguf.clear();
                g_tokenizer_service_model_id = cfg.model_id;
            } else {
                g_tokenizer_service_gguf     = worker_gguf;
                g_tokenizer_service_model_id.clear();
            }
            free_tokenizer_service();
            tokenizer_ready = tokenizer_service_vocab() != nullptr;
            if (!tokenizer_ready) {
                res.status = 500;
                res.set_content(json({
                    { "status", "error" },
                    { "runtime_ready", false },
                    { "error", "tokenizer shell failed to load" },
                    { "worker_gguf", worker_gguf },
                }).dump(), "application/json");
                return;
            }
        } else if (rt_role == "embedding") {
            if (worker_gguf.empty() && bind_ready) {
                g_embedding_service_gguf.clear();
                g_embedding_service_model_id   = cfg.model_id;
                g_embedding_service_layer_start = 0;
                g_embedding_service_layer_end   = 0;
            } else {
                g_embedding_service_gguf     = worker_gguf;
                g_embedding_service_model_id.clear();
            }
            free_embedding_service();
            std::string emb_err;
            if (!ensure_embedding_service_loaded(emb_err)) {
                res.status = 500;
                res.set_content(json({
                    { "status", "error" },
                    { "runtime_ready", false },
                    { "error", emb_err.empty() ? "embedding service failed to load" : emb_err },
                    { "worker_gguf", worker_gguf },
                }).dump(), "application/json");
                return;
            }
        } else if (rt_role == "output_head") {
            if (worker_gguf.empty() && bind_ready) {
                g_output_service_gguf.clear();
                g_output_service_model_id   = cfg.model_id;
                g_output_service_layer_start = 0;
                g_output_service_layer_end   = 0;
            } else {
                g_output_service_gguf     = worker_gguf;
                g_output_service_model_id.clear();
            }
            free_output_service();
            std::string out_err;
            if (!ensure_output_service_loaded(out_err)) {
                res.status = 500;
                res.set_content(json({
                    { "status", "error" },
                    { "runtime_ready", false },
                    { "error", out_err.empty() ? "output service failed to load" : out_err },
                    { "worker_gguf", worker_gguf },
                }).dump(), "application/json");
                return;
            }
        } else if (is_sampler) {
            g_sampler_service_ready = true;
        } else if (cfg.role == DIST_ROLE_ENTRY) {
            if (worker_gguf.empty() && bind_ready) {
                g_entry_worker_gguf.clear();
                g_entry_model_id = cfg.model_id;
            } else {
                g_entry_worker_gguf = worker_gguf;
                g_entry_model_id.clear();
            }
            free_entry_tokenizer();
            tokenizer_ready = entry_node_vocab() != nullptr;
        }

        const runtime_bind_result & bind = resolved.bind;
        const bool tensors_ready = bind.tensors_ready;
        const bool cached_gguf   = resolved.from_cache || bind.cached_gguf_ready;

        const double ms = std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - t0).count();
        res.set_content(json({
            { "status", "ok" },
            { "runtime_ready", true },
            { "worker_gguf", worker_gguf },
            { "tokenizer_ready", tokenizer_ready },
            { "embedding_ready", rt_role == "embedding" ? embedding_service_ready() : false },
            { "output_ready", rt_role == "output_head" ? output_service_ready() : false },
            { "tensors_ready", tensors_ready },
            { "cached_gguf", cached_gguf },
            { "materialize_required", bind.materialize_required },
            { "compat_materialized", resolved.compat_materialized },
            { "bind_source", bind_ready && worker_gguf.empty() ? "layer_store" : "compat_gguf" },
            { "materialization_time_ms", ms },
        }).dump(), "application/json");
    });

    svr.Post("/configure", [agent_bin](const httplib::Request & req, httplib::Response & res) {
        json body;
        try {
            body = json::parse(req.body);
        } catch (...) {
            res.status = 400;
            res.set_content(R"({"error":"invalid json"})", "application/json");
            return;
        }

        perf_consume_trace_from_body(body, "configure");
        const dist_configure_req cfg = parse_configure(body);
        const std::string role_name = dist_role_name(cfg.role);
        perf_session_span startup_span(
                "SESSION_WORKER_STARTUP",
                g_node_id.c_str(),
                role_name.c_str());

        std::string worker_model = g_model_path;
        std::string err;
        const bool skip_materialize = body.value("skip_materialize", false);

        if (dist_debug_enabled() && !cfg.session_id.empty()) {
            dist_debug_set_session(cfg.session_id);
        }

        if (skip_materialize) {
            worker_model = body.value("worker_gguf", "");
            if (worker_model.empty()) {
                if (runtime_layer_first_enabled() && !cfg.model_id.empty()) {
                    worker_model = LAYER_STORE_MODEL_SENTINEL;
                } else {
                    res.status = 400;
                    res.set_content(json({
                        { "ok", false },
                        { "error", "worker_gguf required when skip_materialize=true" },
                    }).dump(), "application/json");
                    return;
                }
            }
        } else if (!cfg.model_id.empty()) {
            runtime_worker_gguf_resolve_result resolved{};
            const std::string materialized =
                    configure_worker_runtime(cfg, body, err, &resolved);
            if (materialized.empty()) {
                if (err.empty() && resolved.bind.tensors_ready && runtime_layer_first_enabled()) {
                    worker_model = LAYER_STORE_MODEL_SENTINEL;
                } else {
                    if (err.empty() && resolved.bind.tensors_ready) {
                        err = "tensors bound but compat GGUF required for worker spawn";
                    }
                    res.status = 500;
                    res.set_content(json({ { "ok", false }, { "error", err } }).dump(), "application/json");
                    return;
                }
            } else {
                worker_model = materialized;
            }
        } else if (worker_model.empty()) {
            res.status = 400;
            res.set_content(json({
                { "ok", false },
                { "error", "model_id or local --model required" },
            }).dump(), "application/json");
            return;
        }

        if (cfg.role == DIST_ROLE_ENTRY) {
            g_external_embedding = body.value("external_embedding", false);
            if (body.contains("embedding_service") && body["embedding_service"].is_object()) {
                const auto & es = body["embedding_service"];
                g_embedding_service_host = es.value("host", "");
                g_embedding_service_port = es.value("port", 0);
            } else if (g_external_embedding && !g_embedding_service_gguf.empty()) {
                g_embedding_service_host.clear();
                g_embedding_service_port = 0;
            }
        } else if (cfg.role == DIST_ROLE_FINAL) {
            g_external_output = body.value("external_output", false);
            if (body.contains("output_service") && body["output_service"].is_object()) {
                const auto & os = body["output_service"];
                g_output_service_host = os.value("host", "");
                g_output_service_port = os.value("port", 0);
            } else if (g_external_output && !g_output_service_gguf.empty()) {
                g_output_service_host.clear();
                g_output_service_port = 0;
            }
        }

        if (!start_worker(agent_bin, cfg, worker_model, err)) {
            res.status = 500;
            res.set_content(json({ { "ok", false }, { "error", err } }).dump(), "application/json");
            return;
        }
        g_runtime_stats.configure_count++;
        g_runtime_stats.runtime_load_count++;

        if (cfg.role == DIST_ROLE_ENTRY) {
            g_pipeline_ctrl_port = cfg.ctrl_port;
            g_pipeline_layer_end   = cfg.layer_end;
            g_pipeline_session_id  = cfg.session_id;
            g_pipeline_protocol    = DIST_RUNTIME_PROTOCOL_V1;
            if (worker_model == LAYER_STORE_MODEL_SENTINEL) {
                g_entry_worker_gguf.clear();
                g_entry_model_id = cfg.model_id;
            } else {
                g_entry_worker_gguf = worker_model;
                g_entry_model_id.clear();
            }
        }

        res.set_content(json({
            { "ok", true },
            { "role", dist_role_name(cfg.role) },
            { "worker_pid", (int) (worker_proc_slot(cfg.role) ? worker_proc_slot(cfg.role)->pid : 0) },
            { "worker_state", read_worker_ready_state_file(
                    worker_state_file_slot(cfg.role) ? *worker_state_file_slot(cfg.role) : "") },
        }).dump(), "application/json");
    });

    svr.Post("/pipeline/generate", [](const httplib::Request & req, httplib::Response & res) {
        json body;
        try {
            body = json::parse(req.body);
        } catch (...) {
            res.status = 400;
            res.set_content(R"({"ok":false,"error":"invalid json"})", "application/json");
            return;
        }

        const int max_tokens = body.value("max_tokens", 16);
        const int layer_end  = body.value("layer_end", g_pipeline_layer_end);
        const std::string trace_id = body.value("trace_id", "");
        const bool perf_trace = body.value("perf_trace", false);
        if (perf_trace) {
            dist_set_env("DIST_PERF_TRACE", "1");
            if (!g_models_dir.empty()) {
                dist_set_env("DIST_PERF_TRACE_DIR", (g_models_dir + "/perf_trace").c_str());
            }
            // node_agent is long-lived; its perf config was cached at startup
            // (likely DIST_PERF_TRACE=0). Re-read so the client decode loop traces.
            perf_trace_reload_config();
            perf_trace_set_node_id(g_node_id);
        }

        std::vector<int32_t> prompt_tokens;
        if (body.contains("prompt_tokens") && body["prompt_tokens"].is_array()) {
            for (const auto & t : body["prompt_tokens"]) {
                prompt_tokens.push_back(t.get<int32_t>());
            }
        }

        const std::string prompt_text = body.value("prompt", "");
        if (prompt_tokens.empty() && !prompt_text.empty()) {
            res.status = 400;
            res.set_content(R"({"ok":false,"error":"prompt_tokens required; use tokenizer service"})", "application/json");
            return;
        }

        if (prompt_tokens.empty()) {
            res.status = 400;
            res.set_content(R"({"ok":false,"error":"prompt or prompt_tokens required"})", "application/json");
            return;
        }
        if (layer_end <= 0) {
            res.status = 503;
            res.set_content(R"({"ok":false,"error":"pipeline not configured"})", "application/json");
            return;
        }

        std::vector<int32_t> out_tokens;
        std::string err;
        json timing = json::object();
        const std::string * trace_ptr = trace_id.empty() ? nullptr : &trace_id;
        if (!run_local_pipeline_generate(
                    prompt_tokens, max_tokens, layer_end, out_tokens, err, &timing, trace_ptr)) {
            res.status = 500;
            res.set_content(json({ { "ok", false }, { "error", err }, { "tokens", json::array() } }).dump(),
                    "application/json");
            return;
        }

        json tokens_json = json::array();
        std::string text;
        for (const int32_t t : out_tokens) {
            tokens_json.push_back(t);
        }
        g_runtime_stats.pipeline_generate_count++;
        res.set_content(json({
            { "ok", true },
            { "tokens", tokens_json },
            { "text", text },
            { "count", out_tokens.size() },
            { "timing", timing },
            { "trace_id", trace_id },
        }).dump(), "application/json");
    });

    svr.Post("/shutdown", [](const httplib::Request &, httplib::Response & res) {
        stop_all_workers();
        free_entry_tokenizer();
        free_tokenizer_service();
        free_embedding_service();
        free_output_service();
        g_entry_worker_gguf.clear();
        g_entry_model_id.clear();
        g_tokenizer_service_gguf.clear();
        g_embedding_service_gguf.clear();
        g_output_service_gguf.clear();
        g_sampler_service_ready = false;
        g_external_embedding = false;
        g_external_output = false;
        g_embedding_service_host.clear();
        g_embedding_service_port = 0;
        g_output_service_host.clear();
        g_output_service_port = 0;
        res.set_content(R"({"ok":true})", "application/json");
    });

    // Model store API
    
    // GET /models/local - List locally installed models
    svr.Get("/models/local", [](const httplib::Request &, httplib::Response & res) {
        json models_json = json::array();

        std::lock_guard<std::mutex> lock(g_model_store_mu);
        auto local_models = g_model_store.get_local_models();
        for (const auto & model : local_models) {
            std::string status = model.ready ? "ready" : "unknown";
            const auto active_it = g_active_downloads.find(model.model_id);
            if (active_it != g_active_downloads.end()) {
                status = "downloading";
            }
            const auto error_it = g_download_errors.find(model.model_id);
            if (error_it != g_download_errors.end()) {
                status = "error";
            }

            json model_json = {
                { "model_id", model.model_id },
                { "local_path", model.local_path },
                { "size_bytes", model.size_bytes },
                { "ready", model.ready },
                { "status", status },
                { "installed_ms", model.installed_ms }
            };
            if (error_it != g_download_errors.end()) {
                model_json["error"] = error_it->second;
            }
            models_json.push_back(model_json);
        }
        
        res.set_content(models_json.dump(), "application/json");
    });

    // GET /installed-layers - Report per-layer state from Layer Store (Task 9.7).
    svr.Get("/installed-layers", [](const httplib::Request & req, httplib::Response & res) {
        const std::string model_id = req.get_param_value("model");
        json layers_json = json::array();

        if (!model_id.empty()) {
            const dist_node_capabilities caps = dist_probe_capabilities();
            const std::string device = dist_normalize_device(caps.gpu_backend, caps.has_gpu);
            const layer_store store = get_layer_store(model_id);

            for (const layer_blob & blob : store.list_layers()) {
                std::string state = "READY";
                if (!store.verify_layer(blob.layer_index, blob.checksum)) {
                    state = "CORRUPTED";
                }

                layers_json.push_back({
                    { "layer", blob.layer_index },
                    { "node", g_node_id },
                    { "node_id", g_node_id },
                    { "device", device },
                    { "size_bytes", blob.size_bytes },
                    { "checksum", blob.checksum },
                    { "state", state },
                });
            }

            for (const layer_store::blob_tensor_info & blob : store.list_blob_tensors()) {
                std::string state = "READY";
                const std::string checksum = blob.checksum.empty()
                        ? ("manifest:tensor:" + blob.tensor_name)
                        : blob.checksum;
                if (!store.verify_blob_tensor(blob.blob_id, blob.tensor_name, checksum)) {
                    state = "CORRUPTED";
                }

                layers_json.push_back({
                    { "layer", -1 },
                    { "layer_index", -1 },
                    { "blob_id", blob.blob_id },
                    { "tensor_name", blob.tensor_name },
                    { "node", g_node_id },
                    { "node_id", g_node_id },
                    { "device", device },
                    { "size_bytes", blob.size_bytes },
                    { "checksum", checksum },
                    { "state", state },
                });
            }
        }

        res.set_content(json({
            { "model", model_id },
            { "layers", layers_json },
        }).dump(), "application/json");
    });

    // POST /models/{model_id}/install/execute - Run Synchronization Engine locally.
    // POST /models/{model_id}/reset - Clear local layer store bytes.
    svr.Get(R"(/models/([^/]+)/workers/entry)", [](const httplib::Request & req, httplib::Response & res) {
        const std::string model_id = req.matches[1];
        layer_store store = get_layer_store(model_id);
        const std::filesystem::path path = store.model_root() / "worker_entry.gguf";
        std::error_code ec;
        if (!std::filesystem::exists(path, ec)) {
            res.status = 404;
            res.set_content(json({ { "error", "worker_entry.gguf not found" } }).dump(), "application/json");
            return;
        }
        std::ifstream in(path, std::ios::binary);
        if (!in) {
            res.status = 500;
            res.set_content(json({ { "error", "failed to read worker_entry.gguf" } }).dump(), "application/json");
            return;
        }
        std::string body((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
        res.set_content(std::move(body), "application/octet-stream");
    });

    svr.Post(R"(/models/([^/]+)/reset)", [](const httplib::Request & req, httplib::Response & res) {
        const std::string model_id = req.matches[1];

        json body;
        try {
            body = req.body.empty() ? json::object() : json::parse(req.body);
        } catch (...) {
            body = json::object();
        }
        const bool keep_manifest = body.value("keep_manifest", true);

        layer_store store = get_layer_store(model_id);
        if (!store.clear_model_storage(keep_manifest)) {
            res.status = 500;
            res.set_content(json({ { "ok", false }, { "error", "clear failed" } }).dump(), "application/json");
            return;
        }

        res.set_content(json({
            { "ok", true },
            { "model_id", model_id },
            { "keep_manifest", keep_manifest },
            { "node_id", g_node_id },
        }).dump(), "application/json");
    });

    svr.Post(R"(/models/([^/]+)/install/execute)", [](const httplib::Request & req, httplib::Response & res) {
        const std::string model_id = req.matches[1];

        json body;
        try {
            body = json::parse(req.body);
        } catch (...) {
            res.status = 400;
            res.set_content(R"({"error":"invalid json"})", "application/json");
            return;
        }

        if (!body.contains("operations") || !body["operations"].is_array()) {
            res.status = 400;
            res.set_content(R"({"error":"operations array required"})", "application/json");
            return;
        }

        const std::string trace_id = body.value("trace_id", "");
        const bool perf_trace = body.value("perf_trace", false);
        if (perf_trace || perf_trace_enabled()) {
            dist_set_env("DIST_PERF_TRACE", "1");
            if (!g_models_dir.empty()) {
                dist_set_env("DIST_PERF_TRACE_DIR", (g_models_dir + "/perf_trace").c_str());
            }
            // See perf_consume_trace_from_body: g_cfg is cached at first use.
            perf_trace_reload_config();
            if (!trace_id.empty()) {
                perf_trace_set_node_id(g_node_id);
                perf_trace_set_component("sync");
                perf_trace_set_context(trace_id, "install", -1);
                perf_gpu_poll_start();
            }
        }

        std::vector<install_operation> operations;
        for (const auto & item : body["operations"]) {
            operations.push_back(install_operation::from_json(item));
        }

        const model_manifest * manifest_ptr = nullptr;
        model_manifest manifest;
        if (body.contains("manifest") && body["manifest"].is_object()) {
            manifest = model_manifest::from_json(body["manifest"]);
            manifest_ptr = &manifest;
        }

        layer_store store = get_layer_store(model_id);
        const std::string job_id = g_sync_engine.start_job(
                model_id, operations, store, manifest_ptr);

        res.set_content(json({
            { "job_id", job_id },
            { "model_id", model_id },
            { "status", "started" },
            { "operation_count", static_cast<int>(operations.size()) },
        }).dump(), "application/json");
    });

    // GET /jobs/{job_id} - Synchronization job progress.
    svr.Get(R"(/jobs/(.+))", [](const httplib::Request & req, httplib::Response & res) {
        const std::string job_id = req.matches[1];
        const auto job = g_sync_engine.get_job(job_id);
        if (!job.has_value()) {
            res.status = 404;
            res.set_content(R"({"error":"job not found"})", "application/json");
            return;
        }
        res.set_content(job->to_json().dump(), "application/json");
    });

    // POST /models/install - Install a model locally
    svr.Post("/models/install", [agent_bin](const httplib::Request & req, httplib::Response & res) {
        json body;
        try {
            body = json::parse(req.body);
        } catch (...) {
            res.status = 400;
            res.set_content(R"({"error":"invalid json"})", "application/json");
            return;
        }
        
        const std::string model_id = body.value("model", "");
        const std::string repo = body.value("repo", "");
        const std::string file = body.value("file", "");
        
        if (model_id.empty() || repo.empty() || file.empty()) {
            res.status = 400;
            res.set_content(R"({"error":"model, repo, and file required"})", "application/json");
            return;
        }
        
        const std::string expected_path = g_model_store.get_model_path(model_id);
        const std::string downloader_script = agent_bin + "../../tools/distributed/download_model.py";

        {
            std::lock_guard<std::mutex> lock(g_model_store_mu);
            const auto * existing = g_model_store.find_model(model_id);
            if (existing && existing->ready && std::filesystem::exists(existing->local_path)) {
                res.set_content(json({
                    { "model_id", model_id },
                    { "status", "already_installed" },
                    { "local_path", existing->local_path }
                }).dump(), "application/json");
                return;
            }

            if (g_active_downloads.find(model_id) != g_active_downloads.end()) {
                res.set_content(json({
                    { "model_id", model_id },
                    { "status", "download_in_progress" },
                    { "local_path", expected_path }
                }).dump(), "application/json");
                return;
            }

            // Fast path: seed from an already-present GGUF (symlink) instead of
            // downloading. Keeps installs offline/fast when the file is local.
            const std::string local_src = find_local_source(file);
            if (!local_src.empty()) {
                std::error_code ec;
                std::filesystem::create_directories(
                        std::filesystem::path(expected_path).parent_path(), ec);
                std::filesystem::remove(expected_path, ec);
                std::filesystem::create_symlink(local_src, expected_path, ec);
                if (ec) {
                    // Fall back to a copy if symlinks are unsupported.
                    ec.clear();
                    std::filesystem::copy_file(local_src, expected_path,
                            std::filesystem::copy_options::overwrite_existing, ec);
                }
                if (!ec && file_size_or_zero(expected_path) > 0) {
                    installed_model seeded;
                    seeded.model_id = model_id;
                    seeded.local_path = expected_path;
                    seeded.size_bytes = file_size_or_zero(expected_path);
                    seeded.ready = true;
                    seeded.installed_ms = now_ms();
                    g_model_store.add_model(seeded);
                    g_download_errors.erase(model_id);
                    g_model_store.save_state(g_models_state_path);
                    res.set_content(json({
                        { "model_id", model_id },
                        { "status", "already_installed" },
                        { "local_path", expected_path }
                    }).dump(), "application/json");
                    return;
                }
            }

            installed_model new_model;
            new_model.model_id = model_id;
            new_model.local_path = expected_path;
            new_model.ready = false;
            new_model.size_bytes = file_size_or_zero(expected_path);
            new_model.installed_ms = 0;

            g_model_store.add_model(new_model);
            g_download_errors.erase(model_id);
            g_active_downloads.insert(model_id);
            g_model_store.save_state(g_models_state_path);
        }

        start_model_download(model_id, repo, file, expected_path, downloader_script);

        res.set_content(json({
            { "model_id", model_id },
            { "status", "download_started" },
            { "local_path", expected_path }
        }).dump(), "application/json");
    });

    const char * model_label = g_model_path.empty() ? "(layer-store)" : g_model_path.c_str();
    fprintf(stderr, "node_agent: listening on %s:%d model=%s score=%.1f\n",
            bind_host.c_str(), http_port, model_label, g_benchmark.score);

    if (!svr.listen(bind_host.c_str(), http_port)) {
        fprintf(stderr, "node_agent: failed to bind %s:%d\n", bind_host.c_str(), http_port);
        stop_all_workers();
        return 1;
    }

    stop_all_workers();
    return 0;
}
