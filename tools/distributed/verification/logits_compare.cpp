#include "logits_compare.h"

#include "split_gen_common.h"
#include "verification_common.h"

#include "ggml-backend.h"
#include "llama.h"

#include <cmath>

static logits_compare_stats stats_for_logits(
        const std::vector<float> & logits,
        const std::vector<float> * other) {
    logits_compare_stats stats;
    stats.n_elements = logits.size();
    if (logits.empty()) {
        return stats;
    }

    for (size_t i = 0; i < logits.size(); ++i) {
        if (logits[i] > stats.max_value) {
            stats.max_value  = logits[i];
            stats.max_index  = (int32_t) i;
        }
    }

    if (other && other->size() == logits.size()) {
        float mae = 0.0f;
        for (size_t i = 0; i < logits.size(); ++i) {
            const float d = std::fabs(logits[i] - (*other)[i]);
            mae += d;
            stats.max_abs_err = std::max(stats.max_abs_err, d);
        }
        stats.mean_abs_err = mae / static_cast<float>(logits.size());
        stats.match        = float_vectors_near(logits.data(), other->data(), logits.size());
    }

    return stats;
}

static bool extract_last_logits(
        const std::string & model_path,
        const std::string & prompt,
        std::vector<float> & logits) {
    logits.clear();

    ggml_backend_load_all();
    llama_model_params mparams = llama_model_default_params();
    mparams.n_gpu_layers       = 0;
    llama_model * model = llama_model_load_from_file(model_path.c_str(), mparams);
    if (!model) {
        return false;
    }

    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx   = 512;
    cparams.n_batch = 512;
    cparams.no_perf = true;

    llama_context * ctx = llama_init_from_model(model, cparams);
    if (!ctx) {
        llama_model_free(model);
        return false;
    }

    const llama_vocab * vocab = llama_model_get_vocab(model);
    const auto tokens         = split_gen_tokenize(vocab, prompt);
    if (tokens.empty()) {
        llama_free(ctx);
        llama_model_free(model);
        return false;
    }

    if (split_gen_decode_tokens(ctx, tokens, 0, true) != 0) {
        llama_free(ctx);
        llama_model_free(model);
        return false;
    }

    const float * raw = llama_get_logits_ith(ctx, (int32_t) tokens.size() - 1);
    const int32_t n_vocab = llama_vocab_n_tokens(vocab);
    if (!raw || n_vocab <= 0) {
        llama_free(ctx);
        llama_model_free(model);
        return false;
    }

    logits.assign(raw, raw + n_vocab);

    llama_free(ctx);
    llama_model_free(model);
    return true;
}

logits_compare_result compare_logits_files(
        const std::string & original_gguf,
        const std::string & materialized_gguf,
        const std::string & prompt) {
    logits_compare_result result;
    result.summary.name = "stage_10_logits";

    if (!extract_last_logits(original_gguf, prompt, result.logits_original)) {
        result.summary.status  = verify_status::fail;
        result.summary.message = "original logits extraction failed";
        return result;
    }
    if (!extract_last_logits(materialized_gguf, prompt, result.logits_materialized)) {
        result.summary.status  = verify_status::fail;
        result.summary.message = "materialized logits extraction failed";
        return result;
    }

    if (result.logits_original.size() != result.logits_materialized.size()) {
        result.summary.status  = verify_status::fail;
        result.summary.message = "logits size mismatch";
        return result;
    }

    result.original     = stats_for_logits(result.logits_original, &result.logits_materialized);
    result.materialized = stats_for_logits(result.logits_materialized, &result.logits_original);

    result.summary.details = {
        { "size", result.original.n_elements },
        { "original_max", result.original.max_value },
        { "original_argmax", result.original.max_index },
        { "materialized_max", result.materialized.max_value },
        { "materialized_argmax", result.materialized.max_index },
        { "mae", result.original.mean_abs_err },
        { "max_error", result.original.max_abs_err },
        { "match", result.original.match },
    };

    if (result.original.match) {
        result.summary.status  = verify_status::ok;
        result.summary.message = "logits match within tolerance";
    } else {
        result.summary.status  = verify_status::fail;
        result.summary.message = "logits differ (mae=" +
                std::to_string(result.original.mean_abs_err) + ")";
    }
    return result;
}
