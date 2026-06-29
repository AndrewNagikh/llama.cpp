#include "trace_recorder.h"

#include "llama.h"

// Observation helpers — no inference changes.

void dist_debug_log_position(
        trace_recorder * rec,
        int32_t step,
        const char * phase,
        llama_context * ctx,
        int32_t position,
        int32_t n_tokens,
        int32_t batch_size = 1);

void dist_debug_log_kv(
        trace_recorder * rec,
        int32_t step,
        const char * phase,
        llama_context * ctx,
        llama_seq_id seq_id = 0);

void dist_debug_log_hidden_out(
        trace_recorder * rec,
        int32_t step,
        const char * phase,
        llama_context * ctx,
        int32_t n_tokens,
        int32_t n_embd,
        const char * dump_tag = nullptr);

void dist_debug_log_logits_out(
        trace_recorder * rec,
        int32_t step,
        const char * phase,
        llama_context * ctx,
        int32_t vocab_size,
        int32_t logits_index = -1);

int32_t dist_debug_sample_or_argmax(
        llama_sampler * smpl,
        llama_context * ctx,
        int32_t logits_index,
        bool * used_argmax);

void dist_debug_log_runtime_state(
        trace_recorder * rec,
        int32_t step,
        const char * phase,
        const char * worker,
        llama_context * ctx,
        int32_t batch_n_tokens,
        const int32_t * token_ids,
        const int32_t * positions,
        int32_t prev_token = -1,
        int32_t current_token = -1);
