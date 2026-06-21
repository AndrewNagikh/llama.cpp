#include "ggml-backend.h"
#include "llama.h"

#include "../src/llama-ext.h"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

struct diff_stats {
    float max_abs = 0.0f;
    float mean_abs = 0.0f;
};

static diff_stats compare_vectors(const std::vector<float> & a, const std::vector<float> & b) {
    diff_stats s;
    if (a.size() != b.size() || a.empty()) {
        return s;
    }

    double sum = 0.0;
    for (size_t i = 0; i < a.size(); ++i) {
        const float d = std::fabs(a[i] - b[i]);
        s.max_abs = std::max(s.max_abs, d);
        sum += d;
    }
    s.mean_abs = (float) (sum / (double) a.size());
    return s;
}

static int decode_tokens(llama_context * ctx, const std::vector<llama_token> & tokens, bool all_outputs) {
    const int32_t n_tokens = (int32_t) tokens.size();
    llama_batch batch = llama_batch_init(n_tokens, 0, 1);
    for (int32_t i = 0; i < n_tokens; ++i) {
        batch.token[i]    = tokens[i];
        batch.pos[i]      = i;
        batch.n_seq_id[i] = 1;
        batch.seq_id[i][0] = 0;
        batch.logits[i]   = all_outputs || (i == n_tokens - 1);
    }
    batch.n_tokens = n_tokens;
    const int ret = llama_decode(ctx, batch);
    llama_batch_free(batch);
    return ret;
}

static int decode_hidden(
        llama_context * ctx,
        const std::vector<float> & hidden,
        int32_t n_tokens,
        int32_t n_embd,
        bool last_output) {
    llama_batch batch = llama_batch_init(n_tokens, n_embd, 1);
    for (int32_t i = 0; i < n_tokens; ++i) {
        batch.pos[i]      = i;
        batch.n_seq_id[i] = 1;
        batch.seq_id[i][0] = 0;
        batch.logits[i]   = last_output && (i == n_tokens - 1);
        std::memcpy(
                batch.embd + (size_t) i * n_embd,
                hidden.data() + (size_t) i * n_embd,
                (size_t) n_embd * sizeof(float));
    }
    batch.n_tokens = n_tokens;
    const int ret = llama_decode(ctx, batch);
    llama_batch_free(batch);
    return ret;
}

static std::vector<float> get_last_logits(llama_context * ctx, int n_vocab) {
    const float * logits = llama_get_logits_ith(ctx, -1);
    return std::vector<float>(logits, logits + n_vocab);
}

int main(int argc, char ** argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: %s MODEL.gguf\n", argv[0]);
        return 1;
    }

    ggml_backend_load_all();

    llama_model * model = llama_model_load_from_file(argv[1], llama_model_default_params());
    if (!model) {
        fprintf(stderr, "failed to load model: %s\n", argv[1]);
        return 1;
    }

    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx = 512;
    cparams.no_perf = true;
    llama_context * ctx = llama_init_from_model(model, cparams);
    if (!ctx) {
        fprintf(stderr, "failed to create context\n");
        return 1;
    }

    const int32_t n_layer = llama_model_n_layer(model);
    const int32_t n_embd  = llama_model_n_embd(model);
    const int32_t half    = n_layer / 2;
    const int32_t n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model));

    const llama_vocab * vocab = llama_model_get_vocab(model);
    const std::string prompt = "Hello";
    const int n_tokens = -llama_tokenize(vocab, prompt.c_str(), prompt.size(), nullptr, 0, true, true);
    std::vector<llama_token> tokens(n_tokens);
    llama_tokenize(vocab, prompt.c_str(), prompt.size(), tokens.data(), tokens.size(), true, true);

    printf("model: n_layer=%d n_embd=%d n_tokens=%d half=%d\n", n_layer, n_embd, n_tokens, half);

    // full forward: layers [0, n_layer)
    llama_set_layer_range(ctx, 0, -1);
    llama_clear_hidden_state(ctx);
    llama_memory_clear(llama_get_memory(ctx), true);
    if (decode_tokens(ctx, tokens, false) != 0) {
        fprintf(stderr, "full forward decode failed\n");
        return 1;
    }
    const auto full_logits = get_last_logits(ctx, n_vocab);
    printf("full forward: logits[0..3]=%f %f %f %f\n",
            full_logits[0], full_logits[1], full_logits[2], full_logits[3]);

    // step 1: layers [0, half) -> hidden state for all tokens
    llama_set_layer_range(ctx, 0, half);
    llama_clear_hidden_state(ctx);
    llama_memory_clear(llama_get_memory(ctx), true);
    if (decode_tokens(ctx, tokens, true) != 0) {
        fprintf(stderr, "step 1 decode failed\n");
        return 1;
    }
    const float * hidden_ptr = llama_get_embeddings(ctx);
    if (hidden_ptr == nullptr) {
        fprintf(stderr, "step 1: expected hidden state\n");
        return 1;
    }
    const std::vector<float> hidden(hidden_ptr, hidden_ptr + (size_t) n_tokens * n_embd);
    printf("step 1: captured hidden state (%d tokens x %d), h[0..3]=%f %f %f %f\n",
            n_tokens, n_embd, hidden[0], hidden[1], hidden[2], hidden[3]);

    // step 2: inject hidden state, layers [half, n_layer) -> logits
    llama_set_layer_range(ctx, half, n_layer);
    llama_memory_clear(llama_get_memory(ctx), true);
    llama_set_hidden_state(ctx, hidden.data(), n_tokens);
    if (decode_hidden(ctx, hidden, n_tokens, n_embd, true) != 0) {
        fprintf(stderr, "step 2 decode failed\n");
        return 1;
    }
    const auto split_logits = get_last_logits(ctx, n_vocab);
    printf("split forward: logits[0..3]=%f %f %f %f\n",
            split_logits[0], split_logits[1], split_logits[2], split_logits[3]);

    const diff_stats diff = compare_vectors(full_logits, split_logits);
    printf("comparison: max_abs_diff=%.9f mean_abs_diff=%.9f\n", diff.max_abs, diff.mean_abs);

    const float tol = 1e-5f;
    if (diff.max_abs < tol) {
        printf("test-injection-proof: OK (max_abs_diff < %.1e)\n", tol);
        llama_free(ctx);
        llama_model_free(model);
        return 0;
    }

    fprintf(stderr, "test-injection-proof: FAILED (max_abs_diff=%.9f >= %.1e)\n", diff.max_abs, tol);
    llama_free(ctx);
    llama_model_free(model);
    return 1;
}
