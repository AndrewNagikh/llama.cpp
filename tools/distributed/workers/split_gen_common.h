#pragma once

#include "llama.h"

#include "llama-distributed.h"

#include <chrono>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

static constexpr int    SPLIT_GEN_SEED          = 1234;
static constexpr int    SPLIT_GEN_MAX_NEW       = 16;
static constexpr int    SPLIT_GEN_LAYER_END     = 8;
static const char *     SPLIT_GEN_PROMPT        = "Tell me a joke";

static bool split_gen_pipe_trace_enabled() {
    const char * v = std::getenv("DIST_PIPE_TRACE");
    return v != nullptr && std::strcmp(v, "0") != 0;
}

static void split_gen_pipe_trace(
        const char * worker,
        const char * phase,
        const char * step,
        int32_t n_tokens,
        int32_t n_embd,
        int32_t pos_start,
        int32_t layer_start,
        int32_t layer_end) {
    if (!split_gen_pipe_trace_enabled()) {
        return;
    }
    const auto now = std::chrono::system_clock::now().time_since_epoch();
    const auto ts_ms = std::chrono::duration_cast<std::chrono::milliseconds>(now).count();
    std::fprintf(stderr,
            "pipe_trace timestamp_ms=%lld worker=%s phase=%s step=%s n_tokens=%d n_embd=%d pos_start=%d layer_start=%d layer_end=%d\n",
            (long long) ts_ms,
            worker,
            phase,
            step,
            n_tokens,
            n_embd,
            pos_start,
            layer_start,
            layer_end);
    std::fflush(stderr);
}

static void split_gen_write_ready_state(const std::string & ready_file, const char * state) {
    if (ready_file.empty()) {
        return;
    }
    FILE * f = std::fopen(ready_file.c_str(), "w");
    if (f == nullptr) {
        return;
    }
    std::fprintf(f, "%s\n", state);
    std::fclose(f);
}

// Sampling settings carried from /session/create down to whichever process
// actually samples (the final pipeline worker, or the output service when the
// output head lives on another node). Defaults reproduce the greedy chain this
// runtime used before the settings were exposed, so anything that does not set
// them explicitly keeps the old deterministic argmax behavior.
struct split_gen_sampler_params {
    float    temp           = 0.0f;   // <= 0 -> greedy (deterministic argmax)
    int32_t  top_k          = 1;      // <= 0 -> disabled (whole vocab)
    float    top_p          = 1.0f;   // 1.0 -> disabled
    float    min_p          = 0.0f;   // 0.0 -> disabled
    float    repeat_penalty = 1.0f;   // 1.0 -> disabled
    int32_t  repeat_last_n  = 64;     // 0 -> disabled, -1 -> context size
    uint32_t seed           = LLAMA_DEFAULT_SEED;

    bool is_greedy() const { return temp <= 0.0f; }
};

// Chain order follows llama.cpp's own convention (penalties -> truncation ->
// temperature -> selection); penalties deliberately come first so the
// repetition scan runs before top-k/top-p have narrowed the candidate set.
static llama_sampler * split_gen_make_sampler_from(const split_gen_sampler_params & p) {
    auto sparams = llama_sampler_chain_default_params();
    sparams.no_perf = true;
    llama_sampler * smpl = llama_sampler_chain_init(sparams);

    if (p.is_greedy()) {
        // Preserved verbatim from the original hardcoded chain.
        llama_sampler_chain_add(smpl, llama_sampler_init_top_k(1));
        llama_sampler_chain_add(smpl, llama_sampler_init_top_p(1.0f, 1));
        llama_sampler_chain_add(smpl, llama_sampler_init_temp(0.0f));
        llama_sampler_chain_add(smpl, llama_sampler_init_greedy());
        return smpl;
    }

    if (p.repeat_penalty != 1.0f && p.repeat_last_n != 0) {
        llama_sampler_chain_add(smpl, llama_sampler_init_penalties(
                p.repeat_last_n, p.repeat_penalty, 0.0f, 0.0f));
    }
    if (p.top_k > 0) {
        llama_sampler_chain_add(smpl, llama_sampler_init_top_k(p.top_k));
    }
    if (p.top_p < 1.0f) {
        llama_sampler_chain_add(smpl, llama_sampler_init_top_p(p.top_p, 1));
    }
    if (p.min_p > 0.0f) {
        llama_sampler_chain_add(smpl, llama_sampler_init_min_p(p.min_p, 1));
    }
    llama_sampler_chain_add(smpl, llama_sampler_init_temp(p.temp));
    llama_sampler_chain_add(smpl, llama_sampler_init_dist(p.seed));
    return smpl;
}

static llama_sampler * split_gen_make_sampler() {
    return split_gen_make_sampler_from(split_gen_sampler_params{});
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
        bool all_outputs,
        bool use_hidden_state_api = true) {
    if (use_hidden_state_api) {
        llama_set_hidden_state(ctx, hidden, n_tokens);
    }
    llama_batch batch = llama_batch_init(n_tokens, n_embd, 1);
    for (int32_t i = 0; i < n_tokens; ++i) {
        batch.pos[i]       = pos_start + i;
        batch.n_seq_id[i]  = 1;
        batch.seq_id[i][0] = 0;
        batch.logits[i]    = all_outputs ? 1 : (i == n_tokens - 1);
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

// layer_start > 0: hidden injection must run one token at a time so KV is
// updated per position (matches split_gen3_b middle-stage prefill).
static int split_gen_decode_hidden_seq(
        llama_context * ctx,
        const float * hidden,
        int32_t n_tokens,
        int32_t n_embd,
        llama_pos pos_start) {
    for (int32_t i = 0; i < n_tokens; ++i) {
        const float * in = hidden + (size_t) i * (size_t) n_embd;
        if (split_gen_decode_hidden(ctx, in, 1, n_embd, pos_start + i, true) != 0) {
            return -1;
        }
    }
    return 0;
}

// Task 19 verify waves can leave a speculative KV tail on stages that
// processed tokens the verifier later rejected. Every stage tracks its next
// expected position; a wave arriving below it truncates the stale tail first.
static void split_gen_rollback_kv(llama_context * ctx, const int32_t next_pos,
        const int32_t pos_start, const char * tag) {
    if (pos_start < next_pos) {
        llama_memory_seq_rm(llama_get_memory(ctx), 0, pos_start, -1);
        fprintf(stderr, "%s: kv rollback [%d, %d) -> %d\n", tag, pos_start, next_pos, pos_start);
    }
}

static std::vector<llama_token> split_gen_tokenize(const llama_vocab * vocab, const std::string & prompt) {
    const int n = -llama_tokenize(vocab, prompt.c_str(), prompt.size(), nullptr, 0, true, true);
    std::vector<llama_token> tokens(n);
    llama_tokenize(vocab, prompt.c_str(), prompt.size(), tokens.data(), tokens.size(), true, true);
    return tokens;
}
