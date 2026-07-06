#include "hidden_pipeline_parity.h"

#include "runtime_debug/tensor_stats.h"
#include "split_gen_common.h"
#include "ggml-backend.h"
#include "llama.h"

#include <cmath>

static double relative_diff(const double lhs, const double rhs) {
    const double denom = std::max(std::fabs(lhs), std::fabs(rhs));
    if (denom == 0.0) {
        return 0.0;
    }
    return std::fabs(lhs - rhs) / denom;
}

static bool aggregate_stats_match(const tensor_stats & ref, const tensor_stats & producer) {
    if (ref.n_elements != producer.n_elements || ref.has_nan != producer.has_nan ||
            ref.has_inf != producer.has_inf) {
        return false;
    }
    constexpr double mean_tol = 5e-4;
    constexpr double std_rel_tol = 5e-3;
    constexpr double l2_rel_tol = 5e-3;
    return std::fabs(static_cast<double>(ref.mean) - producer.mean) <= mean_tol &&
            relative_diff(ref.stddev, producer.stddev) <= std_rel_tol &&
            relative_diff(ref.l2_norm, producer.l2_norm) <= l2_rel_tol;
}

hidden_pipeline_parity_result verify_hidden_at_layer_boundary(
        const std::string & model_path,
        const std::string & prompt,
        int32_t layer_end,
        const parsed_trace & entry_trace) {
    hidden_pipeline_parity_result result;

    const trace_event * entry_hidden = nullptr;
    for (const auto & ev : entry_trace.events) {
        if (ev.event == "hidden" && ev.phase == "prefill" && ev.has_hidden) {
            entry_hidden = &ev;
            break;
        }
    }
    if (entry_hidden == nullptr) {
        result.message = "entry trace missing prefill hidden event";
        return result;
    }

    ggml_backend_load_all();
    llama_model * model = llama_model_load_from_file(model_path.c_str(), llama_model_default_params());
    if (!model) {
        result.message = "model load failed";
        return result;
    }

    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx        = 512;
    cparams.n_batch      = 512;
    cparams.no_perf      = true;
    cparams.embeddings   = true;
    llama_context * ctx = llama_init_from_model(model, cparams);
    if (!ctx) {
        llama_model_free(model);
        result.message = "context init failed";
        return result;
    }
    llama_set_embeddings(ctx, true);
    llama_set_layer_range(ctx, 0, layer_end);

    const llama_vocab * vocab = llama_model_get_vocab(model);
    const auto tokens         = split_gen_tokenize(vocab, prompt);
    if (tokens.empty() || split_gen_decode_tokens(ctx, tokens, 0, true) != 0) {
        llama_free(ctx);
        llama_model_free(model);
        result.message = "mono decode failed";
        return result;
    }

    const int32_t n_embd = llama_model_n_embd(model);
    const int64_t expected_elements =
            entry_hidden->hidden_stats.n_elements > 0 ?
                    entry_hidden->hidden_stats.n_elements :
                    static_cast<int64_t>(entry_hidden->n_tokens) * entry_hidden->n_embd;
    if (expected_elements <= 0) {
        llama_free(ctx);
        llama_model_free(model);
        result.message = "producer trace hidden shape missing";
        return result;
    }

    const int64_t full_prefill_elements = static_cast<int64_t>(tokens.size()) * n_embd;
    const float * hidden = nullptr;
    int64_t hidden_elements = 0;
    if (expected_elements == full_prefill_elements) {
        hidden = llama_get_embeddings(ctx);
        hidden_elements = full_prefill_elements;
    } else if (expected_elements == n_embd) {
        hidden = llama_get_embeddings_ith(ctx, (int32_t) tokens.size() - 1);
        hidden_elements = n_embd;
    } else {
        llama_free(ctx);
        llama_model_free(model);
        result.message = "producer trace hidden shape is not comparable with reference boundary";
        return result;
    }
    if (hidden == nullptr) {
        llama_free(ctx);
        llama_model_free(model);
        result.message = "mono hidden missing";
        return result;
    }

    const tensor_stats mono_stats = compute_tensor_stats(hidden, hidden_elements);
    result.reference_stats = mono_stats;
    result.producer_stats = entry_hidden->hidden_stats;
    result.exact_sha_match = (mono_stats.sha256 == entry_hidden->hidden_stats.sha256);
    result.aggregate_match = aggregate_stats_match(mono_stats, entry_hidden->hidden_stats);
    result.metrics.max_abs_err =
            std::fabs(static_cast<double>(mono_stats.mean) - entry_hidden->hidden_stats.mean);
    result.metrics.mean_abs_err = result.metrics.max_abs_err;
    result.metrics.l2_diff = std::fabs(mono_stats.l2_norm - entry_hidden->hidden_stats.l2_norm);
    result.metrics.cosine_sim = result.aggregate_match ? 1.0 : 0.0;
    result.metrics.match = result.exact_sha_match || result.aggregate_match;
    result.ok = result.metrics.match;
    result.message = result.ok ? "hidden match at layer boundary"
                               : "hidden stats differ at layer " + std::to_string(layer_end);

    llama_free(ctx);
    llama_model_free(model);
    return result;
}
