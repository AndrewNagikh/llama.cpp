#pragma once

#include "llama.h"

#include "../src/llama-ext.h"

#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

static constexpr int    SPLIT_GEN_SEED          = 1234;
static constexpr int    SPLIT_GEN_MAX_NEW       = 16;
static constexpr int    SPLIT_GEN_LAYER_END     = 8;
static const char *     SPLIT_GEN_PROMPT        = "Tell me a joke";

static llama_sampler * split_gen_make_sampler() {
    auto sparams = llama_sampler_chain_default_params();
    sparams.no_perf = true;
    llama_sampler * smpl = llama_sampler_chain_init(sparams);
    llama_sampler_chain_add(smpl, llama_sampler_init_top_k(1));
    llama_sampler_chain_add(smpl, llama_sampler_init_top_p(1.0f, 1));
    llama_sampler_chain_add(smpl, llama_sampler_init_temp(0.0f));
    llama_sampler_chain_add(smpl, llama_sampler_init_greedy());
    return smpl;
}

static std::string split_gen_token_text(const llama_vocab * vocab, llama_token id) {
    char buf[256];
    const int n = llama_token_to_piece(vocab, id, buf, sizeof(buf), 0, true);
    if (n < 0) {
        return {};
    }
    return std::string(buf, n);
}

static int split_gen_decode_tokens(
        llama_context * ctx,
        const std::vector<llama_token> & tokens,
        llama_pos pos_start,
        bool all_outputs) {
    const int32_t n_tokens = (int32_t) tokens.size();
    llama_batch batch = llama_batch_init(n_tokens, 0, 1);
    for (int32_t i = 0; i < n_tokens; ++i) {
        batch.token[i]     = tokens[i];
        batch.pos[i]       = pos_start + i;
        batch.n_seq_id[i]  = 1;
        batch.seq_id[i][0] = 0;
        batch.logits[i]    = all_outputs ? 1 : (i == n_tokens - 1);
    }
    batch.n_tokens = n_tokens;
    const int ret = llama_decode(ctx, batch);
    llama_batch_free(batch);
    return ret;
}

static int split_gen_decode_one(llama_context * ctx, llama_token tok, llama_pos pos) {
    llama_batch batch = llama_batch_init(1, 0, 1);
    batch.token[0]     = tok;
    batch.pos[0]       = pos;
    batch.n_seq_id[0]  = 1;
    batch.seq_id[0][0] = 0;
    batch.logits[0]    = 1;
    batch.n_tokens     = 1;
    const int ret = llama_decode(ctx, batch);
    llama_batch_free(batch);
    return ret;
}

static int split_gen_decode_hidden(
        llama_context * ctx,
        const float * hidden,
        int32_t n_tokens,
        int32_t n_embd,
        llama_pos pos_start,
        bool last_output) {
    llama_batch batch = llama_batch_init(n_tokens, n_embd, 1);
    for (int32_t i = 0; i < n_tokens; ++i) {
        batch.pos[i]       = pos_start + i;
        batch.n_seq_id[i]  = 1;
        batch.seq_id[i][0] = 0;
        batch.logits[i]    = last_output && (i == n_tokens - 1);
        std::memcpy(
                batch.embd + (size_t) i * n_embd,
                hidden + (size_t) i * n_embd,
                (size_t) n_embd * sizeof(float));
    }
    batch.n_tokens = n_tokens;
    const int ret = llama_decode(ctx, batch);
    llama_batch_free(batch);
    return ret;
}

static std::vector<llama_token> split_gen_tokenize(const llama_vocab * vocab, const std::string & prompt) {
    const int n = -llama_tokenize(vocab, prompt.c_str(), prompt.size(), nullptr, 0, true, true);
    std::vector<llama_token> tokens(n);
    llama_tokenize(vocab, prompt.c_str(), prompt.size(), tokens.data(), tokens.size(), true, true);
    return tokens;
}
