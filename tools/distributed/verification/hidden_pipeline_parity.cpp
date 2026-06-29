#include "hidden_pipeline_parity.h"

#include "runtime_debug/tensor_stats.h"
#include "split_gen_common.h"
#include "ggml-backend.h"
#include "llama.h"

#include <cmath>

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
    const float * hidden = llama_get_embeddings_ith(ctx, (int32_t) tokens.size() - 1);
    if (hidden == nullptr) {
        llama_free(ctx);
        llama_model_free(model);
        result.message = "mono hidden missing";
        return result;
    }

    const tensor_stats mono_stats = compute_tensor_stats(hidden, n_embd);
    result.metrics.max_abs_err = std::fabs(mono_stats.mean - entry_hidden->hidden_stats.mean);
    result.metrics.match       = (mono_stats.sha256 == entry_hidden->hidden_stats.sha256);
    result.ok                  = result.metrics.match;
    result.message             = result.ok ? "hidden match at layer boundary"
                                             : "hidden sha256 differs at layer " + std::to_string(layer_end);

    llama_free(ctx);
    llama_model_free(model);
    return result;
}
