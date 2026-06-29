#pragma once

#include "llama.h"

#include <cstdint>
#include <string>
#include <vector>

// Task 9.8.5 — runtime state snapshot after decode (observation only).

struct runtime_state_snapshot {
    int32_t step           = -1;
    std::string phase;
    std::string worker;
    int32_t n_past         = 0;
    int32_t batch_n_tokens = 0;
    int32_t seq_id         = 0;
    std::vector<int32_t> token_ids;
    std::vector<int32_t> positions;
    int32_t kv_entries     = 0;
    int32_t kv_n_layer     = 0;
    int32_t hidden_n_tokens = 0;
    int32_t n_embd         = 0;
    const void * hidden_ptr = nullptr;
    size_t hidden_bytes    = 0;
    std::string hidden_sha256;
    int32_t layer_start    = 0;
    int32_t layer_end      = 0;
    int32_t prev_token     = -1;
    int32_t current_token  = -1;
};

bool dist_debug_runtime_state_enabled();

runtime_state_snapshot capture_runtime_state(
        llama_context * ctx,
        int32_t step,
        const char * phase,
        const char * worker,
        int32_t batch_n_tokens,
        const int32_t * token_ids,
        const int32_t * positions,
        int32_t seq_id = 0,
        int32_t prev_token = -1,
        int32_t current_token = -1);

void dist_debug_emit_runtime_state(
        class trace_recorder * rec,
        const runtime_state_snapshot & snap);

std::string runtime_state_json(const runtime_state_snapshot & snap);
