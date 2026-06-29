#include "layer_runner.h"

#include "split_gen_common.h"
#include "llama-distributed.h"

#include "../../../src/llama-ext.h"

#include "ggml-backend.h"

#include <cstring>

layer_runner_ctx layer_runner_load(const std::string & model_path, const int32_t n_ctx) {
    layer_runner_ctx rt{};
    ggml_backend_load_all();

    llama_model * model = llama_model_load_from_file(model_path.c_str(), llama_model_default_params());
    if (!model) {
        return rt;
    }

    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx      = n_ctx;
    cparams.n_batch    = n_ctx;
    cparams.no_perf    = true;
    cparams.embeddings = true;

    llama_context * ctx = llama_init_from_model(model, cparams);
    if (!ctx) {
        llama_model_free(model);
        return rt;
    }

    llama_set_embeddings(ctx, true);
    rt.model = model;
    rt.ctx   = ctx;
    return rt;
}

void layer_runner_free(layer_runner_ctx & rt) {
    if (rt.ctx) {
        llama_free(rt.ctx);
        rt.ctx = nullptr;
    }
    if (rt.model) {
        llama_model_free(rt.model);
        rt.model = nullptr;
    }
}

std::vector<llama_token> layer_runner_tokenize(const layer_runner_ctx & rt, const std::string & prompt) {
    if (!rt.model) {
        return {};
    }
    const llama_vocab * vocab = llama_model_get_vocab(rt.model);
    return split_gen_tokenize(vocab, prompt);
}

layer_boundary_sample layer_runner_capture_boundary(
        layer_runner_ctx & rt,
        const std::vector<llama_token> & tokens,
        const int32_t layer_index,
        const bool capture_input) {
    layer_boundary_sample sample{};
    sample.layer_index = layer_index;

    if (!rt.model || !rt.ctx || tokens.empty() || layer_index < 0) {
        return sample;
    }

    const int32_t n_layer = llama_model_n_layer(rt.model);
    if (layer_index >= n_layer) {
        return sample;
    }

    const int32_t n_embd = llama_model_n_embd(rt.model);
    sample.n_embd        = n_embd;
    sample.token_index   = (int32_t) tokens.size() - 1;

    llama_memory_clear(llama_get_memory(rt.ctx), true);
    llama_clear_hidden_state(rt.ctx);

    for (int32_t il = 0; il < n_layer; ++il) {
        llama_set_embeddings_layer_inp(rt.ctx, (uint32_t) il, false);
    }
    if (capture_input) {
        llama_set_embeddings_layer_inp(rt.ctx, (uint32_t) layer_index, true);
    }

    llama_set_embeddings(rt.ctx, true);
    llama_set_layer_range(rt.ctx, 0, layer_index + 1);

    sample.positions.resize(tokens.size());
    for (size_t i = 0; i < tokens.size(); ++i) {
        sample.positions[i] = (int32_t) i;
    }

    if (split_gen_decode_tokens(rt.ctx, tokens, 0, true) != 0) {
        return sample;
    }

    const float * out = llama_get_embeddings_ith(rt.ctx, sample.token_index);
    if (out != nullptr) {
        sample.output_stats = compute_tensor_stats(out, n_embd);
        sample.output.assign(out, out + n_embd);
        sample.has_output = true;
    }

    if (capture_input) {
        const float * inp = llama_get_embeddings_layer_inp(rt.ctx, (uint32_t) layer_index);
        if (inp != nullptr) {
            const float * inp_last = inp + (size_t) sample.token_index * n_embd;
            sample.input_stats     = compute_tensor_stats(inp_last, n_embd);
            sample.input.assign(inp_last, inp_last + n_embd);
            sample.has_input = true;
        }
    }

    llama_memory_t mem = llama_get_memory(rt.ctx);
    sample.kv_seq_max  = mem ? llama_memory_seq_pos_max(mem, 0) : -1;

    return sample;
}
