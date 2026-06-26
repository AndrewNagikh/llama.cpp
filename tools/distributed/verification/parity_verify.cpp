#include "parity_verify.h"

#include "split_gen_common.h"
#include "verification_common.h"

#include "ggml-backend.h"
#include "llama.h"

#include <cmath>

static std::vector<llama_token> tokenize_prompt(llama_model * model, const std::string & prompt) {
    const llama_vocab * vocab = llama_model_get_vocab(model);
    return split_gen_tokenize(vocab, prompt);
}

static bool run_greedy(
        const std::string & model_path,
        const std::string & prompt,
        const int max_tokens,
        std::vector<llama_token> & out_tokens,
        std::vector<float> * last_logits) {
    out_tokens.clear();
    if (last_logits) {
        last_logits->clear();
    }

    ggml_backend_load_all();
    llama_model_params mparams = llama_model_default_params();
    mparams.n_gpu_layers = 0;
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

    llama_sampler * smpl = split_gen_make_sampler();
    const auto prompt_tokens = tokenize_prompt(model, prompt);
    if (prompt_tokens.empty()) {
        llama_sampler_free(smpl);
        llama_free(ctx);
        llama_model_free(model);
        return false;
    }

    if (split_gen_decode_tokens(ctx, prompt_tokens, 0, true) != 0) {
        llama_sampler_free(smpl);
        llama_free(ctx);
        llama_model_free(model);
        return false;
    }

    if (last_logits) {
        const float * logits = llama_get_logits_ith(ctx, (int32_t) prompt_tokens.size() - 1);
        const int32_t n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model));
        if (logits) {
            last_logits->assign(logits, logits + n_vocab);
        }
    }

    llama_token cur = llama_sampler_sample(smpl, ctx, -1);
    llama_sampler_accept(smpl, cur);
    out_tokens.push_back(cur);

    for (int i = 1; i < max_tokens; ++i) {
        if (split_gen_decode_one(ctx, cur, (llama_pos) (prompt_tokens.size() + i - 1)) != 0) {
            break;
        }
        cur = llama_sampler_sample(smpl, ctx, -1);
        llama_sampler_accept(smpl, cur);
        out_tokens.push_back(cur);
    }

    llama_sampler_free(smpl);
    llama_free(ctx);
    llama_model_free(model);
    return true;
}

parity_result verify_inference_parity(
        const std::string & original_gguf,
        const std::string & materialized_gguf,
        const std::string & prompt,
        const int max_tokens) {
    parity_result result;
    result.logits.name    = "logits";
    result.sampling.name  = "sampling";

    std::vector<float> logits_orig;
    std::vector<float> logits_mat;
    std::vector<llama_token> toks_orig;
    std::vector<llama_token> toks_mat;

    if (!run_greedy(original_gguf, prompt, max_tokens, toks_orig, &logits_orig)) {
        result.logits.status   = verify_status::fail;
        result.logits.message  = "original model inference failed";
        result.sampling.status = verify_status::skip;
        return result;
    }
    if (!run_greedy(materialized_gguf, prompt, max_tokens, toks_mat, &logits_mat)) {
        result.logits.status   = verify_status::fail;
        result.logits.message  = "materialized model inference failed";
        result.sampling.status = verify_status::skip;
        return result;
    }

    for (const auto t : toks_orig) {
        result.tokens_original.push_back((int32_t) t);
    }
    for (const auto t : toks_mat) {
        result.tokens_materialized.push_back((int32_t) t);
    }

    if (logits_orig.empty() || logits_mat.empty() || logits_orig.size() != logits_mat.size()) {
        result.logits.status  = verify_status::fail;
        result.logits.message = "logits size mismatch";
    } else if (float_vectors_near(logits_orig.data(), logits_mat.data(), logits_orig.size())) {
        result.logits.status  = verify_status::ok;
        result.logits.message = "last-token logits match within tolerance";
    } else {
        result.logits.status  = verify_status::fail;
        result.logits.message = "logits differ beyond tolerance";
    }

    const size_t n = std::min(toks_orig.size(), toks_mat.size());
    bool sample_ok = (toks_orig.size() == toks_mat.size());
    for (size_t i = 0; i < n && sample_ok; ++i) {
        if (toks_orig[i] != toks_mat[i]) {
            sample_ok = false;
        }
    }
    result.sampling.status  = sample_ok ? verify_status::ok : verify_status::fail;
    result.sampling.message = sample_ok ? "greedy tokens match" : "greedy tokens differ";
    result.sampling.details = {
        { "original", result.tokens_original },
        { "materialized", result.tokens_materialized },
    };
    return result;
}
