#include "debug_hooks.h"

#include "runtime_debug.h"
#include "runtime_state.h"

#include <algorithm>

void dist_debug_log_position(
        trace_recorder * rec,
        const int32_t step,
        const char * phase,
        llama_context * ctx,
        const int32_t position,
        const int32_t n_tokens,
        const int32_t batch_size) {
    if (rec == nullptr || ctx == nullptr) {
        return;
    }
    llama_memory_t mem = llama_get_memory(ctx);
    const llama_pos seq_max = mem ? llama_memory_seq_pos_max(mem, 0) : -1;
    const int32_t seq_len   = seq_max >= 0 ? static_cast<int32_t>(seq_max + 1) : n_tokens;
    rec->emit_position(step, phase, position, n_tokens, seq_len, batch_size);
}

void dist_debug_log_kv(
        trace_recorder * rec,
        const int32_t step,
        const char * phase,
        llama_context * ctx,
        const llama_seq_id seq_id) {
    if (rec == nullptr || ctx == nullptr) {
        return;
    }
    llama_memory_t mem = llama_get_memory(ctx);
    if (!mem) {
        return;
    }
    const llama_pos seq_max = llama_memory_seq_pos_max(mem, seq_id);
    const int32_t kv_entries = seq_max >= 0 ? static_cast<int32_t>(seq_max + 1) : 0;
    const int32_t n_layer    = llama_model_n_layer(llama_get_model(ctx));
    rec->emit_kv(step, phase, kv_entries, n_layer, kv_entries);
}

void dist_debug_log_hidden_out(
        trace_recorder * rec,
        const int32_t step,
        const char * phase,
        llama_context * ctx,
        const int32_t n_tokens,
        const int32_t n_embd,
        const char * dump_tag) {
    if (rec == nullptr || ctx == nullptr || n_tokens <= 0) {
        return;
    }
    const float * hidden = llama_get_embeddings(ctx);
    if (hidden == nullptr) {
        return;
    }
    rec->emit_hidden(step, phase, hidden, n_tokens, n_embd, dump_tag);
}

void dist_debug_log_logits_out(
        trace_recorder * rec,
        const int32_t step,
        const char * phase,
        llama_context * ctx,
        const int32_t vocab_size,
        const int32_t logits_index) {
    if (rec == nullptr || ctx == nullptr || vocab_size <= 0) {
        return;
    }
    const int32_t idx = logits_index >= 0 ? logits_index : -1;
    const float * logits = llama_get_logits_ith(ctx, idx);
    if (logits == nullptr) {
        return;
    }
    rec->emit_logits(step, phase, logits, vocab_size, dist_debug_skip_sampler());
}

int32_t dist_debug_sample_or_argmax(
        llama_sampler * smpl,
        llama_context * ctx,
        const int32_t logits_index,
        bool * used_argmax) {
    if (used_argmax) {
        *used_argmax = false;
    }
    if (dist_debug_skip_sampler()) {
        const llama_vocab * vocab = llama_model_get_vocab(llama_get_model(ctx));
        const int32_t n_vocab     = llama_vocab_n_tokens(vocab);
        const float * logits      = llama_get_logits_ith(ctx, logits_index);
        if (logits == nullptr || n_vocab <= 0) {
            return -1;
        }
        int32_t best = 0;
        for (int32_t i = 1; i < n_vocab; ++i) {
            if (logits[i] > logits[best]) {
                best = i;
            }
        }
        if (used_argmax) {
            *used_argmax = true;
        }
        return best;
    }
    const llama_token tok = llama_sampler_sample(smpl, ctx, logits_index);
    return static_cast<int32_t>(tok);
}

void dist_debug_log_runtime_state(
        trace_recorder * rec,
        const int32_t step,
        const char * phase,
        const char * worker,
        llama_context * ctx,
        const int32_t batch_n_tokens,
        const int32_t * token_ids,
        const int32_t * positions,
        const int32_t prev_token,
        const int32_t current_token) {
    if (rec == nullptr || !dist_debug_runtime_state_enabled()) {
        return;
    }
    const runtime_state_snapshot snap = capture_runtime_state(
            ctx, step, phase, worker, batch_n_tokens, token_ids, positions,
            0, prev_token, current_token);
    dist_debug_emit_runtime_state(rec, snap);
}
