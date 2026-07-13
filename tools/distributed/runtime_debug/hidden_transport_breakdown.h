#pragma once

#include "llama.h"
#include "split_tcp_wire.h"

#include <cstdint>
#include <vector>

struct hidden_pack_gather_stats {
    int64_t alloc_us       = 0;
    int64_t sync_us        = 0;
    int64_t gather_us      = 0;
    int64_t copy_us        = 0;
    int64_t serialize_us   = 0;
    size_t  capacity_before = 0;
    size_t  capacity_after  = 0;
    size_t  size_before     = 0;
    size_t  size_after      = 0;
    bool    capacity_grew   = false;
    int32_t copy_count      = 0;
    int32_t payload_bytes   = 0;
};

struct hidden_pack_send_stats {
    int64_t frame_us          = 0;
    int64_t send_hdr_us       = 0;
    int64_t send_payload_us   = 0;
    int64_t send_us           = 0;
};

struct hidden_pack_stats {
    hidden_pack_gather_stats gather;
    hidden_pack_send_stats   send;
    int64_t pack_total_us = 0;
};

// DIST_RUNTIME_GATHER_SYNC_SPLIT=1: one explicit llama_synchronize before the
// per-token llama_get_embeddings loop (Task 15.2 Option A/B). Default off.
bool hidden_gather_sync_split_enabled();

bool hidden_pack_gather_stage_hidden(
        llama_context * ctx,
        int32_t n_tokens,
        int32_t n_embd,
        std::vector<float> & out,
        hidden_pack_gather_stats & stats);

bool hidden_pack_send_ab_hidden(
        int fd,
        int32_t n_tokens,
        int32_t n_embd,
        int32_t layer_end,
        int32_t pos_start,
        int32_t include_logits,
        const float * data,
        hidden_pack_send_stats & stats);

void hidden_pack_emit_breakdown_spans(
        const hidden_pack_stats & stats,
        int32_t token_idx,
        int32_t payload_bytes);
