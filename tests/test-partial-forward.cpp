#include "ggml-backend.h"
#include "llama.h"

#include "../src/llama-ext.h"

#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

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

static int decode_hidden(llama_context * ctx, const std::vector<float> & hidden, int32_t n_tokens, int32_t n_embd) {
    llama_batch batch = llama_batch_init(n_tokens, n_embd, 1);
    for (int32_t i = 0; i < n_tokens; ++i) {
        batch.pos[i]      = i;
        batch.n_seq_id[i] = 1;
        batch.seq_id[i][0] = 0;
        batch.logits[i]   = (i == n_tokens - 1);
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

static std::vector<llama_token> tokenize_prompt(const llama_vocab * vocab) {
    const std::string prompt = "Hello";
    const int n_tokens = -llama_tokenize(vocab, prompt.c_str(), prompt.size(), nullptr, 0, true, true);
    std::vector<llama_token> tokens(n_tokens);
    llama_tokenize(vocab, prompt.c_str(), prompt.size(), tokens.data(), tokens.size(), true, true);
    return tokens;
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

    const int n_layer = llama_model_n_layer(model);
    const int n_embd  = llama_model_n_embd(model);
    const int half    = n_layer / 2;
    const auto tokens = tokenize_prompt(llama_model_get_vocab(model));
    const int n_tokens = (int) tokens.size();

    // case 1: first half -> hidden state
    llama_set_layer_range(ctx, 0, half);
    llama_clear_hidden_state(ctx);
    llama_memory_clear(llama_get_memory(ctx), true);
    if (decode_tokens(ctx, tokens, true) != 0) {
        fprintf(stderr, "decode failed for layer_start=0 layer_end=%d\n", half);
        return 1;
    }
    const float * hidden_ptr = llama_get_embeddings(ctx);
    if (hidden_ptr == nullptr) {
        fprintf(stderr, "expected non-null embeddings for partial forward\n");
        return 1;
    }
    const std::vector<float> hidden(hidden_ptr, hidden_ptr + (size_t) n_tokens * n_embd);
    printf("case layer_start=0 layer_end=%d: embedding[0..3]=%f %f %f %f\n",
            half, hidden[0], hidden[1], hidden[2], hidden[3]);

    // case 2: inject hidden, second half -> logits
    llama_set_layer_range(ctx, half, n_layer);
    llama_memory_clear(llama_get_memory(ctx), true);
    llama_set_hidden_state(ctx, hidden.data(), n_tokens);
    if (decode_hidden(ctx, hidden, n_tokens, n_embd) != 0) {
        fprintf(stderr, "decode failed for layer_start=%d layer_end=%d\n", half, n_layer);
        return 1;
    }
    const float * split_logits = llama_get_logits(ctx);
    if (split_logits == nullptr) {
        fprintf(stderr, "expected non-null logits for injected second half\n");
        return 1;
    }
    printf("case layer_start=%d layer_end=%d: logits[0..3]=%f %f %f %f\n",
            half, n_layer, split_logits[0], split_logits[1], split_logits[2], split_logits[3]);

    // case 3: full forward
    llama_set_layer_range(ctx, 0, -1);
    llama_clear_hidden_state(ctx);
    llama_memory_clear(llama_get_memory(ctx), true);
    if (decode_tokens(ctx, tokens, false) != 0) {
        fprintf(stderr, "decode failed for full forward\n");
        return 1;
    }
    const float * full_logits = llama_get_logits(ctx);
    if (full_logits == nullptr) {
        fprintf(stderr, "expected non-null logits for full forward\n");
        return 1;
    }
    printf("case layer_start=0 layer_end=-1: logits[0..3]=%f %f %f %f\n",
            full_logits[0], full_logits[1], full_logits[2], full_logits[3]);

    const int n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model));
    const auto logits_a = std::vector<float>(full_logits, full_logits + n_vocab);
    const auto logits_b = std::vector<float>(split_logits, split_logits + n_vocab);
    if (logits_a != logits_b) {
        fprintf(stderr, "logits mismatch between full and split+inject forward\n");
        return 1;
    }

    llama_memory_clear(llama_get_memory(ctx), true);
    llama_set_layer_range(ctx, 0, -1);
    decode_tokens(ctx, tokens, false);
    const auto logits_c = std::vector<float>(llama_get_logits(ctx), llama_get_logits(ctx) + n_vocab);
    llama_memory_clear(llama_get_memory(ctx), true);
    llama_set_layer_range(ctx, 0, -1);
    decode_tokens(ctx, tokens, false);
    const auto logits_d = std::vector<float>(llama_get_logits(ctx), llama_get_logits(ctx) + n_vocab);
    if (logits_c != logits_d) {
        fprintf(stderr, "logits mismatch between default and explicit full forward\n");
        return 1;
    }

    printf("test-partial-forward: OK (n_layer=%d n_embd=%d)\n", n_layer, n_embd);
    llama_free(ctx);
    llama_model_free(model);
    return 0;
}
