#pragma once

// Stable public API for distributed inference (partial forward + hidden state injection).
// Distributed runtime tools must include only this header — not src/llama-ext.h.

#include "llama.h"

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// Partial forward: execute only layers in [start, end). end < 0 means n_layer().
LLAMA_API void llama_set_layer_range(struct llama_context * ctx, int32_t start, int32_t end);

// Hidden state input for partial forward when layer_start > 0.
// data layout: [n_embd * n_tokens], row-major per token (same as batch.embd).
// Copied into context; caller may free data after the call.
LLAMA_API void llama_set_hidden_state(struct llama_context * ctx, const float * data, int32_t n_tokens);

// Read back hidden state previously set via llama_set_hidden_state().
// Returns number of floats copied into out (0 if none). out may be NULL to query size only.
LLAMA_API int32_t llama_get_hidden_state_n_tokens(struct llama_context * ctx);
LLAMA_API int32_t llama_get_hidden_state(struct llama_context * ctx, float * out, int32_t out_nfloats);

// Clear hidden state set by llama_set_hidden_state().
LLAMA_API void llama_clear_hidden_state(struct llama_context * ctx);

#ifdef __cplusplus
}
#endif
