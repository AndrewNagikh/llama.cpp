#include "final_runtime_verify.h"

#include "split_gen_common.h"
#include "runtime_debug/hidden_transport.h"
#include "verification/verification_common.h"

#include "ggml-backend.h"
#include "llama-distributed.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <sstream>

namespace {

static void json_escape(std::ostringstream & os, const std::string & s) {
    os << '"';
    for (char c : s) {
        if (c == '"') {
            os << "\\\"";
        } else if (c == '\\') {
            os << "\\\\";
        } else {
            os << c;
        }
    }
    os << '"';
}

static float logits_entropy(const float * logits, const int32_t n_vocab) {
    if (logits == nullptr || n_vocab <= 0) {
        return 0.0f;
    }
    float max_l = logits[0];
    for (int32_t i = 1; i < n_vocab; ++i) {
        max_l = std::max(max_l, logits[i]);
    }
    double sum = 0.0;
    for (int32_t i = 0; i < n_vocab; ++i) {
        sum += std::exp(static_cast<double>(logits[i] - max_l));
    }
    if (sum <= 0.0) {
        return 0.0f;
    }
    double ent = 0.0;
    for (int32_t i = 0; i < n_vocab; ++i) {
        const double p = std::exp(static_cast<double>(logits[i] - max_l)) / sum;
        if (p > 1e-12) {
            ent -= p * std::log(p);
        }
    }
    return static_cast<float>(ent);
}

static int32_t logits_argmax(const float * logits, const int32_t n_vocab) {
    if (logits == nullptr || n_vocab <= 0) {
        return -1;
    }
    int32_t best_i = 0;
    float best_v   = logits[0];
    for (int32_t i = 1; i < n_vocab; ++i) {
        if (logits[i] > best_v) {
            best_v = logits[i];
            best_i = i;
        }
    }
    return best_i;
}

static std::vector<std::pair<int32_t, float>> logits_top_k(
        const float * logits,
        const int32_t n_vocab,
        const int32_t k) {
    std::vector<std::pair<int32_t, float>> items;
    items.reserve(static_cast<size_t>(n_vocab));
    for (int32_t i = 0; i < n_vocab; ++i) {
        items.emplace_back(i, logits[i]);
    }
    const int32_t kk = std::min(k, n_vocab);
    std::partial_sort(
            items.begin(),
            items.begin() + kk,
            items.end(),
            [](const auto & a, const auto & b) { return a.second > b.second; });
    items.resize(static_cast<size_t>(kk));
    return items;
}

static llama_context_params worker_context_params() {
    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx   = 512;
    cparams.n_batch = 512;
    cparams.no_perf = true;
    return cparams;
}

static llama_context_params mono_hidden_context_params() {
    llama_context_params cparams = worker_context_params();
    cparams.embeddings = true;
    return cparams;
}

static llama_context_params mono_full_context_params() {
    return worker_context_params();
}

static kv_ownership_diag capture_kv_diag(llama_context * ctx, const char * phase) {
    kv_ownership_diag diag{};
    diag.phase = phase ? phase : "";
    if (!ctx) {
        return diag;
    }

    llama_memory_t mem = llama_get_memory(ctx);
    diag.kv_ptr        = reinterpret_cast<uintptr_t>(mem);
    diag.kv_n_layer    = llama_model_n_layer(llama_get_model(ctx));
    if (mem) {
        const llama_pos seq_max = llama_memory_seq_pos_max(mem, 0);
        diag.kv_entries         = seq_max >= 0 ? static_cast<int32_t>(seq_max + 1) : 0;
        diag.n_past             = diag.kv_entries;
    }
    return diag;
}

static hidden_consumption_diag verify_hidden_consumption(
        llama_context * ctx,
        const float * hidden,
        const int32_t n_tokens,
        const int32_t n_embd) {
    hidden_consumption_diag diag{};
    diag.hidden_bytes = static_cast<size_t>(n_tokens) * static_cast<size_t>(n_embd) * sizeof(float);

    if (!ctx || hidden == nullptr || n_tokens <= 0 || n_embd <= 0) {
        diag.message = "invalid hidden consumption inputs";
        return diag;
    }

    diag.sha256_before = sha256_hex(
            reinterpret_cast<const uint8_t *>(hidden),
            diag.hidden_bytes);
    diag.hidden_ptr_before = reinterpret_cast<uintptr_t>(hidden);

    const hidden_state_api_check api = verify_hidden_state_roundtrip(ctx, hidden, n_tokens, n_embd);
    diag.api_roundtrip_ok = api.ok;

    llama_set_hidden_state(ctx, hidden, n_tokens);
    if (split_gen_decode_hidden(ctx, hidden, n_tokens, n_embd, 0, true) != 0) {
        diag.message = "decode_hidden failed during consumption check";
        return diag;
    }

    std::vector<float> roundtrip(static_cast<size_t>(n_tokens) * static_cast<size_t>(n_embd));
    const int32_t got = llama_get_hidden_state(ctx, roundtrip.data(), (int32_t) roundtrip.size());
    if (got > 0) {
        const size_t ncopy = std::min(roundtrip.size(), static_cast<size_t>(got));
        diag.sha256_after  = sha256_hex(
                reinterpret_cast<const uint8_t *>(roundtrip.data()),
                ncopy * sizeof(float));
        diag.hidden_ptr_after = reinterpret_cast<uintptr_t>(roundtrip.data());
        diag.hidden_modified  = diag.sha256_before != diag.sha256_after;
    }

    diag.message = diag.hidden_modified ? "hidden modified after first compute graph"
                                        : "hidden unchanged after first compute graph";
    return diag;
}

static std::string logits_snapshot_json(const logits_snapshot & snap) {
    std::ostringstream os;
    os << "{"
       << "\"sha256\":";
    json_escape(os, snap.stats.sha256);
    os << ",\"argmax\":" << snap.argmax
       << ",\"entropy\":" << snap.entropy
       << ",\"mean\":" << snap.stats.mean
       << ",\"stddev\":" << snap.stats.stddev
       << ",\"l2_norm\":" << snap.stats.l2_norm
       << ",\"top10\":[";
    for (size_t i = 0; i < snap.top10.size(); ++i) {
        if (i > 0) {
            os << ',';
        }
        os << "{\"id\":" << snap.top10[i].first << ",\"logit\":" << snap.top10[i].second << "}";
    }
    os << "]}";
    return os.str();
}

static std::string parity_json(const parity_metrics & m) {
    std::ostringstream os;
    os << "{"
       << "\"match\":" << (m.match ? "true" : "false")
       << ",\"max_abs_err\":" << m.max_abs_err
       << ",\"mean_abs_err\":" << m.mean_abs_err
       << ",\"l2_diff\":" << m.l2_diff
       << ",\"cosine_sim\":" << m.cosine_sim
       << "}";
    return os.str();
}

} // namespace

logits_snapshot capture_logits_snapshot(const float * logits, const int32_t n_vocab) {
    logits_snapshot snap{};
    if (logits == nullptr || n_vocab <= 0) {
        return snap;
    }
    snap.data.assign(logits, logits + n_vocab);
    snap.stats   = compute_tensor_stats(snap.data.data(), n_vocab);
    snap.argmax  = logits_argmax(logits, n_vocab);
    snap.entropy = logits_entropy(logits, n_vocab);
    snap.top10   = logits_top_k(logits, n_vocab, 10);
    return snap;
}

static bool capture_mono_hidden_at_boundary(
        llama_context * ctx,
        const std::vector<llama_token> & tokens,
        const int32_t layer_boundary,
        std::vector<float> & hidden_out) {
    hidden_out.clear();
    if (!ctx || tokens.empty() || layer_boundary <= 0) {
        return false;
    }

    llama_memory_clear(llama_get_memory(ctx), true);
    llama_clear_hidden_state(ctx);
    llama_set_embeddings(ctx, true);
    llama_set_layer_range(ctx, 0, layer_boundary);

    if (split_gen_decode_tokens(ctx, tokens, 0, true) != 0) {
        return false;
    }

    const float * h = llama_get_embeddings(ctx);
    if (h == nullptr) {
        return false;
    }

    const int32_t n_embd = llama_model_n_embd(llama_get_model(ctx));
    hidden_out.assign(h, h + static_cast<size_t>(tokens.size()) * static_cast<size_t>(n_embd));
    return true;
}

static bool capture_mono_full_logits(
        llama_context * ctx,
        const std::vector<llama_token> & tokens,
        std::vector<float> & logits_out) {
    logits_out.clear();
    if (!ctx || tokens.empty()) {
        return false;
    }

    const int32_t n_layer = llama_model_n_layer(llama_get_model(ctx));
    llama_memory_clear(llama_get_memory(ctx), true);
    llama_clear_hidden_state(ctx);
    llama_set_layer_range(ctx, 0, n_layer);

    if (split_gen_decode_tokens(ctx, tokens, 0, false) != 0) {
        return false;
    }

    const float * logits = llama_get_logits_ith(ctx, static_cast<int32_t>(tokens.size()) - 1);
    const int32_t n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(llama_get_model(ctx)));
    if (logits == nullptr || n_vocab <= 0) {
        return false;
    }

    logits_out.assign(logits, logits + n_vocab);
    return true;
}

static bool run_worker_final_prefill(
        llama_context * ctx,
        const float * hidden,
        const int32_t n_tokens,
        const int32_t n_embd,
        const int32_t layer_boundary,
        const int32_t n_layer,
        std::vector<float> & logits_out) {
    logits_out.clear();
    if (!ctx || hidden == nullptr || n_tokens <= 0) {
        return false;
    }

    llama_memory_clear(llama_get_memory(ctx), true);
    llama_clear_hidden_state(ctx);
    llama_set_layer_range(ctx, layer_boundary, n_layer);
    llama_set_hidden_state(ctx, hidden, n_tokens);

    if (split_gen_decode_hidden(ctx, hidden, n_tokens, n_embd, 0, true) != 0) {
        return false;
    }

    const float * logits = llama_get_logits_ith(ctx, n_tokens - 1);
    const int32_t n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(llama_get_model(ctx)));
    if (logits == nullptr || n_vocab <= 0) {
        return false;
    }

    logits_out.assign(logits, logits + n_vocab);
    return true;
}

static bool capture_boundary_hidden_for_token(
        llama_context * ctx,
        const llama_token token,
        const llama_pos pos,
        const int32_t layer_boundary,
        std::vector<float> & hidden_out) {
    hidden_out.clear();
    if (!ctx || layer_boundary <= 0) {
        return false;
    }

    llama_set_embeddings(ctx, true);
    llama_set_layer_range(ctx, 0, layer_boundary);

    if (split_gen_decode_one(ctx, token, pos) != 0) {
        return false;
    }

    const float * h = llama_get_embeddings_ith(ctx, 0);
    if (h == nullptr) {
        h = llama_get_embeddings(ctx);
    }
    if (h == nullptr) {
        return false;
    }

    const int32_t n_embd = llama_model_n_embd(llama_get_model(ctx));
    hidden_out.assign(h, h + n_embd);
    return true;
}

final_runtime_report verify_final_runtime(const final_runtime_config & cfg) {
    final_runtime_report report{};
    report.message = "verify_final_runtime failed";

    if (cfg.mono_path.empty() || cfg.worker_final_path.empty() || cfg.prompt.empty()) {
        report.message = "missing mono_path, worker_final_path, or prompt";
        return report;
    }

    ggml_backend_load_all();

    llama_model * mono_model = llama_model_load_from_file(cfg.mono_path.c_str(), llama_model_default_params());
    llama_model * worker_model = llama_model_load_from_file(cfg.worker_final_path.c_str(), llama_model_default_params());
    if (!mono_model || !worker_model) {
        report.message = "model load failed";
        if (mono_model) {
            llama_model_free(mono_model);
        }
        if (worker_model) {
            llama_model_free(worker_model);
        }
        return report;
    }

    const int32_t n_layer = llama_model_n_layer(mono_model);
    const int32_t n_embd  = llama_model_n_embd(mono_model);
    const int32_t n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(mono_model));
    const int32_t layer_boundary = cfg.layer_boundary > 0 ? cfg.layer_boundary : (n_layer * 2) / 3;

    report.layer_boundary = layer_boundary;
    report.n_embd         = n_embd;

    const llama_vocab * vocab = llama_model_get_vocab(mono_model);
    const auto tokens         = split_gen_tokenize(vocab, cfg.prompt);
    report.n_tokens           = static_cast<int32_t>(tokens.size());
    if (tokens.empty()) {
        report.message = "tokenization failed";
        llama_model_free(mono_model);
        llama_model_free(worker_model);
        return report;
    }

    llama_context * mono_hidden_ctx = llama_init_from_model(mono_model, mono_hidden_context_params());
    llama_context * mono_full_ctx   = llama_init_from_model(mono_model, mono_full_context_params());
    llama_context * worker_ctx      = llama_init_from_model(worker_model, worker_context_params());
    if (!mono_hidden_ctx || !mono_full_ctx || !worker_ctx) {
        report.message = "context init failed";
        llama_free(mono_hidden_ctx);
        llama_free(mono_full_ctx);
        llama_free(worker_ctx);
        llama_model_free(mono_model);
        llama_model_free(worker_model);
        return report;
    }

    llama_set_embeddings(mono_hidden_ctx, true);

    std::vector<float> hidden;
    if (!capture_mono_hidden_at_boundary(mono_hidden_ctx, tokens, layer_boundary, hidden)) {
        report.message = "mono hidden capture failed";
        llama_free(mono_hidden_ctx);
        llama_free(mono_full_ctx);
        llama_free(worker_ctx);
        llama_model_free(mono_model);
        llama_model_free(worker_model);
        return report;
    }

    report.hidden_sha256 = sha256_hex(
            reinterpret_cast<const uint8_t *>(hidden.data()),
            hidden.size() * sizeof(float));

    if (!cfg.output_dir.empty()) {
        std::error_code ec;
        std::filesystem::create_directories(cfg.output_dir, ec);
        report.hidden_bin_path = cfg.output_dir + "/hidden_layer" + std::to_string(layer_boundary) + ".bin";
        if (cfg.save_hidden_bin) {
            hidden_write_bin(report.hidden_bin_path, hidden.data(), hidden.size());
        }
    }

    std::vector<float> mono_logits;
    if (!capture_mono_full_logits(mono_full_ctx, tokens, mono_logits)) {
        report.message = "mono full logits capture failed";
        llama_free(mono_hidden_ctx);
        llama_free(mono_full_ctx);
        llama_free(worker_ctx);
        llama_model_free(mono_model);
        llama_model_free(worker_model);
        return report;
    }

    report.mono_prefill_logits = capture_logits_snapshot(mono_logits.data(), n_vocab);

    report.hidden_consumption = verify_hidden_consumption(
            worker_ctx, hidden.data(), report.n_tokens, n_embd);

    std::vector<float> worker_logits;
    if (!run_worker_final_prefill(
                worker_ctx, hidden.data(), report.n_tokens, n_embd, layer_boundary, n_layer, worker_logits)) {
        report.message = "worker final prefill failed";
        llama_free(mono_hidden_ctx);
        llama_free(mono_full_ctx);
        llama_free(worker_ctx);
        llama_model_free(mono_model);
        llama_model_free(worker_model);
        return report;
    }

    report.worker_prefill_logits = capture_logits_snapshot(worker_logits.data(), n_vocab);
    report.prefill_parity        = compare_tensors(mono_logits.data(), worker_logits.data(), n_vocab);
    report.prefill_logits_pass   = report.prefill_parity.match &&
                                   report.mono_prefill_logits.argmax == report.worker_prefill_logits.argmax;

    report.kv_after_prefill = capture_kv_diag(worker_ctx, "prefill");

    report.context_params = compare_context_params(mono_full_ctx, worker_ctx);
    report.decode_graph   = verify_decode_graph(
            mono_full_ctx, worker_ctx, layer_boundary, n_layer, 1, "decode");
    report.prefill_graph  = verify_decode_graph(
            mono_full_ctx, worker_ctx, layer_boundary, n_layer,
            static_cast<uint32_t>(tokens.size()), "prefill");
    report.mono_cache     = diagnose_runtime_graph_cache(
            mono_full_ctx, layer_boundary, n_layer, 1);
    report.worker_cache   = diagnose_runtime_graph_cache(
            worker_ctx, layer_boundary, n_layer, 1);

    if (!cfg.output_dir.empty()) {
        write_graph_summary_json(cfg.output_dir + "/decode_graph_monolithic.json", report.decode_graph.mono);
        write_graph_summary_json(cfg.output_dir + "/decode_graph_worker.json", report.decode_graph.worker);
    }

    // Decode step equivalence: replay mono full ctx state and compare per-step logits.
    llama_memory_clear(llama_get_memory(mono_full_ctx), true);
    llama_clear_hidden_state(mono_full_ctx);
    llama_set_layer_range(mono_full_ctx, 0, n_layer);
    split_gen_decode_tokens(mono_full_ctx, tokens, 0, false);

    llama_memory_clear(llama_get_memory(mono_hidden_ctx), true);
    llama_clear_hidden_state(mono_hidden_ctx);
    llama_set_embeddings(mono_hidden_ctx, true);
    llama_set_layer_range(mono_hidden_ctx, 0, layer_boundary);
    split_gen_decode_tokens(mono_hidden_ctx, tokens, 0, true);

    llama_memory_clear(llama_get_memory(worker_ctx), true);
    llama_clear_hidden_state(worker_ctx);
    llama_set_layer_range(worker_ctx, layer_boundary, n_layer);
    llama_set_hidden_state(worker_ctx, hidden.data(), report.n_tokens);
    split_gen_decode_hidden(worker_ctx, hidden.data(), report.n_tokens, n_embd, 0, true);

    llama_pos pos       = static_cast<llama_pos>(tokens.size());
    llama_token next_tok = static_cast<llama_token>(report.mono_prefill_logits.argmax);

    for (int32_t step = 0; step < cfg.max_decode_steps; ++step) {
        decode_step_equivalence row{};
        row.step  = step;
        row.phase = "decode";

        if (split_gen_decode_one(mono_full_ctx, next_tok, pos) != 0) {
            row.message = "mono decode failed";
            report.decode_steps.push_back(row);
            break;
        }

        const float * mono_step_logits = llama_get_logits_ith(mono_full_ctx, -1);
        row.mono                       = capture_logits_snapshot(mono_step_logits, n_vocab);

        std::vector<float> step_hidden;
        if (!capture_boundary_hidden_for_token(
                    mono_hidden_ctx, next_tok, pos, layer_boundary, step_hidden)) {
            row.message = "boundary hidden for decode token failed";
            report.decode_steps.push_back(row);
            if (report.first_fail_decode_step < 0) {
                report.first_fail_decode_step = step;
                report.root_cause_field       = "boundary_hidden";
            }
            break;
        }

        llama_set_hidden_state(worker_ctx, step_hidden.data(), 1);
        if (split_gen_decode_hidden(worker_ctx, step_hidden.data(), 1, n_embd, pos, true) != 0) {
            row.message = "worker decode failed";
            report.decode_steps.push_back(row);
            if (report.first_fail_decode_step < 0) {
                report.first_fail_decode_step = step;
                report.root_cause_field       = "worker_decode";
            }
            break;
        }

        const float * worker_step_logits = llama_get_logits_ith(worker_ctx, -1);
        row.worker                       = capture_logits_snapshot(worker_step_logits, n_vocab);
        row.parity                       = compare_tensors(mono_step_logits, worker_step_logits, n_vocab);
        row.pass                         = row.parity.match && row.mono.argmax == row.worker.argmax;
        row.message                      = row.pass ? "PASS" : "logits differ";

        report.decode_steps.push_back(row);
        if (!row.pass && report.first_fail_decode_step < 0) {
            report.first_fail_decode_step = step;
            report.root_cause_field       = "decode_logits";
        }

        next_tok = static_cast<llama_token>(row.mono.argmax);
        ++pos;
    }

    report.all_pass = report.prefill_logits_pass && report.first_fail_decode_step < 0;

    if (!report.prefill_logits_pass) {
        report.root_cause_field = "prefill_logits";
        report.message          = "FIRST DIFF: prefill logits mono argmax=" +
                std::to_string(report.mono_prefill_logits.argmax) +
                " entropy=" + std::to_string(report.mono_prefill_logits.entropy) +
                " worker argmax=" + std::to_string(report.worker_prefill_logits.argmax) +
                " entropy=" + std::to_string(report.worker_prefill_logits.entropy);
    } else if (report.first_fail_decode_step >= 0) {
        report.message = "FIRST DIFF: decode step " + std::to_string(report.first_fail_decode_step) +
                " field=" + report.root_cause_field;
    } else if (!report.context_params.match) {
        report.root_cause_field = "context_params";
        report.message          = report.context_params.message;
    } else if (!report.decode_graph.comparison.match) {
        report.root_cause_field = "decode_graph";
        report.message          = report.decode_graph.comparison.message;
    } else {
        report.message = "Final runtime equivalent to monolithic (prefill + decode steps)";
    }

    llama_free(mono_hidden_ctx);
    llama_free(mono_full_ctx);
    llama_free(worker_ctx);
    llama_model_free(mono_model);
    llama_model_free(worker_model);
    return report;
}

final_logits_report verify_final_logits(const final_logits_config & cfg) {
    final_logits_report report{};

    std::vector<uint8_t> bytes;
    if (!read_file_bytes(cfg.hidden_bin_path, bytes) || bytes.empty()) {
        report.message = "hidden bin read failed";
        return report;
    }
    if (bytes.size() % sizeof(float) != 0) {
        report.message = "hidden bin size misaligned";
        return report;
    }

    std::vector<float> hidden(bytes.size() / sizeof(float));

    ggml_backend_load_all();

    llama_model * mono_model = llama_model_load_from_file(cfg.mono_path.c_str(), llama_model_default_params());
    llama_model * worker_model = llama_model_load_from_file(cfg.worker_final_path.c_str(), llama_model_default_params());
    if (!mono_model || !worker_model) {
        report.message = "model load failed";
        if (mono_model) {
            llama_model_free(mono_model);
        }
        if (worker_model) {
            llama_model_free(worker_model);
        }
        return report;
    }

    const int32_t n_layer = llama_model_n_layer(mono_model);
    const int32_t n_embd  = llama_model_n_embd(mono_model);
    const int32_t n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(mono_model));
    const int32_t layer_boundary = cfg.layer_boundary > 0 ? cfg.layer_boundary : (n_layer * 2) / 3;
    const int32_t n_tokens       = static_cast<int32_t>(hidden.size()) / n_embd;

    if (n_tokens <= 0 || static_cast<size_t>(n_tokens) * static_cast<size_t>(n_embd) != hidden.size()) {
        report.message = "hidden bin size does not match n_embd";
        llama_model_free(mono_model);
        llama_model_free(worker_model);
        return report;
    }

    std::memcpy(hidden.data(), bytes.data(), bytes.size());

    llama_context * mono_full_ctx = llama_init_from_model(mono_model, mono_full_context_params());
    llama_context * worker_ctx    = llama_init_from_model(worker_model, worker_context_params());
    if (!mono_full_ctx || !worker_ctx) {
        report.message = "context init failed";
        llama_free(mono_full_ctx);
        llama_free(worker_ctx);
        llama_model_free(mono_model);
        llama_model_free(worker_model);
        return report;
    }

    const llama_vocab * vocab = llama_model_get_vocab(mono_model);
    const auto tokens         = split_gen_tokenize(vocab, cfg.prompt);
    if (static_cast<int32_t>(tokens.size()) != n_tokens) {
        report.message = "hidden n_tokens != prompt token count";
        llama_free(mono_full_ctx);
        llama_free(worker_ctx);
        llama_model_free(mono_model);
        llama_model_free(worker_model);
        return report;
    }

    std::vector<float> mono_logits;
    if (!capture_mono_full_logits(mono_full_ctx, tokens, mono_logits)) {
        report.message = "mono full logits capture failed";
        llama_free(mono_full_ctx);
        llama_free(worker_ctx);
        llama_model_free(mono_model);
        llama_model_free(worker_model);
        return report;
    }

    std::vector<float> worker_logits;
    if (!run_worker_final_prefill(
                worker_ctx, hidden.data(), n_tokens, n_embd, layer_boundary, n_layer, worker_logits)) {
        report.message = "worker final prefill from hidden bin failed";
        llama_free(mono_full_ctx);
        llama_free(worker_ctx);
        llama_model_free(mono_model);
        llama_model_free(worker_model);
        return report;
    }

    report.mono   = capture_logits_snapshot(mono_logits.data(), n_vocab);
    report.worker = capture_logits_snapshot(worker_logits.data(), n_vocab);
    report.parity = compare_tensors(mono_logits.data(), worker_logits.data(), n_vocab);
    report.pass   = report.parity.match && report.mono.argmax == report.worker.argmax;
    report.message = report.pass
            ? "final logits match monolithic"
            : "final logits differ argmax mono=" + std::to_string(report.mono.argmax) +
                      " worker=" + std::to_string(report.worker.argmax);

    llama_free(mono_full_ctx);
    llama_free(worker_ctx);
    llama_model_free(mono_model);
    llama_model_free(worker_model);
    return report;
}

std::string final_logits_report_json(const final_logits_report & report) {
    std::ostringstream os;
    os << "{"
       << "\"pass\":" << (report.pass ? "true" : "false")
       << ",\"message\":";
    json_escape(os, report.message);
    os << ",\"mono\":" << logits_snapshot_json(report.mono)
       << ",\"worker\":" << logits_snapshot_json(report.worker)
       << ",\"parity\":" << parity_json(report.parity)
       << "}";
    return os.str();
}

std::string final_runtime_report_json(const final_runtime_report & report) {
    std::ostringstream os;
    os << "{"
       << "\"all_pass\":" << (report.all_pass ? "true" : "false")
       << ",\"prefill_logits_pass\":" << (report.prefill_logits_pass ? "true" : "false")
       << ",\"layer_boundary\":" << report.layer_boundary
       << ",\"n_tokens\":" << report.n_tokens
       << ",\"n_embd\":" << report.n_embd
       << ",\"hidden_sha256\":";
    json_escape(os, report.hidden_sha256);
    os << ",\"hidden_bin_path\":";
    json_escape(os, report.hidden_bin_path);
    os << ",\"root_cause_field\":";
    json_escape(os, report.root_cause_field);
    os << ",\"message\":";
    json_escape(os, report.message);
    os << ",\"mono_prefill_logits\":" << logits_snapshot_json(report.mono_prefill_logits)
       << ",\"worker_prefill_logits\":" << logits_snapshot_json(report.worker_prefill_logits)
       << ",\"prefill_parity\":" << parity_json(report.prefill_parity)
       << ",\"context_params\":" << context_params_diff_json(report.context_params)
       << ",\"decode_graph\":" << decode_graph_diff_json(report.decode_graph)
       << ",\"prefill_graph\":" << decode_graph_diff_json(report.prefill_graph)
       << ",\"mono_cache\":" << runtime_cache_diag_json(report.mono_cache)
       << ",\"worker_cache\":" << runtime_cache_diag_json(report.worker_cache)
       << ",\"hidden_consumption\":{"
       << "\"sha256_before\":";
    json_escape(os, report.hidden_consumption.sha256_before);
    os << ",\"sha256_after\":";
    json_escape(os, report.hidden_consumption.sha256_after);
    os << ",\"hidden_bytes\":" << report.hidden_consumption.hidden_bytes
       << ",\"stride\":" << report.hidden_consumption.stride
       << ",\"dtype\":\"" << report.hidden_consumption.dtype << "\""
       << ",\"hidden_ptr_before\":" << report.hidden_consumption.hidden_ptr_before
       << ",\"hidden_ptr_after\":" << report.hidden_consumption.hidden_ptr_after
       << ",\"hidden_modified\":" << (report.hidden_consumption.hidden_modified ? "true" : "false")
       << ",\"api_roundtrip_ok\":" << (report.hidden_consumption.api_roundtrip_ok ? "true" : "false")
       << ",\"message\":";
    json_escape(os, report.hidden_consumption.message);
    os << "}"
       << ",\"kv_after_prefill\":{"
       << "\"kv_ptr\":" << report.kv_after_prefill.kv_ptr
       << ",\"kv_entries\":" << report.kv_after_prefill.kv_entries
       << ",\"kv_n_layer\":" << report.kv_after_prefill.kv_n_layer
       << ",\"seq_count\":" << report.kv_after_prefill.seq_count
       << ",\"n_past\":" << report.kv_after_prefill.n_past
       << ",\"phase\":";
    json_escape(os, report.kv_after_prefill.phase);
    os << "}"
       << ",\"first_fail_decode_step\":" << report.first_fail_decode_step
       << ",\"decode_steps\":[";
    for (size_t i = 0; i < report.decode_steps.size(); ++i) {
        if (i > 0) {
            os << ',';
        }
        const auto & s = report.decode_steps[i];
        os << "{"
           << "\"step\":" << s.step
           << ",\"phase\":";
        json_escape(os, s.phase);
        os << ",\"pass\":" << (s.pass ? "true" : "false")
           << ",\"message\":";
        json_escape(os, s.message);
        os << ",\"mono\":" << logits_snapshot_json(s.mono)
           << ",\"worker\":" << logits_snapshot_json(s.worker)
           << ",\"parity\":" << parity_json(s.parity)
           << "}";
    }
    os << "]}";
    return os.str();
}
