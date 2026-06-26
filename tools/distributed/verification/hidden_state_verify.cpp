#include "hidden_state_verify.h"

#include "split_gen_common.h"
#include "verification_common.h"

#include "ggml-backend.h"
#include "llama.h"

#include <cmath>

static bool extract_last_prompt_hidden(
        const std::string & model_path,
        const std::string & prompt,
        std::vector<float> & hidden) {
    hidden.clear();

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
    cparams.embeddings = true;

    llama_context * ctx = llama_init_from_model(model, cparams);
    if (!ctx) {
        llama_model_free(model);
        return false;
    }
    llama_set_embeddings(ctx, true);

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

    const int32_t idx = (int32_t) tokens.size() - 1;
    const float * emb = llama_get_embeddings_ith(ctx, idx);
    const int32_t n_embd = llama_model_n_embd(model);
    if (!emb || n_embd <= 0) {
        llama_free(ctx);
        llama_model_free(model);
        return false;
    }

    hidden.assign(emb, emb + n_embd);

    llama_free(ctx);
    llama_model_free(model);
    return true;
}

static hidden_state_stats compute_stats(
        const std::vector<float> & v,
        const std::vector<float> * other) {
    hidden_state_stats stats;
    stats.n_elements = v.size();
    if (v.empty()) {
        return stats;
    }

    stats.sha256 = sha256_hex(
            reinterpret_cast<const uint8_t *>(v.data()),
            v.size() * sizeof(float));

    float sum = 0.0f;
    for (const float x : v) {
        const float a = std::fabs(x);
        sum += a;
        stats.max_abs = std::max(stats.max_abs, a);
    }
    stats.mean_abs = sum / static_cast<float>(v.size());

    if (other && other->size() == v.size()) {
        float mae = 0.0f;
        for (size_t i = 0; i < v.size(); ++i) {
            const float d = std::fabs(v[i] - (*other)[i]);
            mae += d;
            stats.max_err = std::max(stats.max_err, d);
        }
        stats.mae_vs_other = mae / static_cast<float>(v.size());
        stats.match        = float_vectors_near(v.data(), other->data(), v.size());
    }

    return stats;
}

hidden_state_compare_result compare_hidden_states(
        const std::string & original_gguf,
        const std::string & materialized_gguf,
        const std::string & prompt) {
    hidden_state_compare_result result;
    result.summary.name = "stage_9_hidden_state";

    std::vector<float> h_orig;
    std::vector<float> h_mat;
    if (!extract_last_prompt_hidden(original_gguf, prompt, h_orig)) {
        result.summary.status  = verify_status::fail;
        result.summary.message = "original hidden state extraction failed";
        return result;
    }
    if (!extract_last_prompt_hidden(materialized_gguf, prompt, h_mat)) {
        result.summary.status  = verify_status::fail;
        result.summary.message = "materialized hidden state extraction failed";
        return result;
    }

    if (h_orig.size() != h_mat.size()) {
        result.summary.status  = verify_status::fail;
        result.summary.message = "hidden state size mismatch";
        return result;
    }

    result.original     = compute_stats(h_orig, &h_mat);
    result.materialized = compute_stats(h_mat, &h_orig);

    result.summary.details = {
        { "original_sha256", result.original.sha256 },
        { "materialized_sha256", result.materialized.sha256 },
        { "mae", result.original.mae_vs_other },
        { "max_error", result.original.max_err },
        { "sha256_match", result.original.sha256 == result.materialized.sha256 },
    };

    if (result.original.match) {
        result.summary.status  = verify_status::ok;
        result.summary.message = "hidden states match within tolerance";
    } else {
        result.summary.status  = verify_status::fail;
        result.summary.message = "hidden states differ (mae=" +
                std::to_string(result.original.mae_vs_other) + ")";
    }
    return result;
}
