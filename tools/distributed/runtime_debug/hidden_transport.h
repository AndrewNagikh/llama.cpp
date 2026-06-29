#pragma once

#include "llama.h"

#include <cstdint>
#include <string>

struct hidden_transport_trace {
    int32_t step           = -1;
    std::string phase;
    std::string direction; // "send" | "recv"
    std::string link;      // "ab" | "bc"
    int32_t n_tokens       = 0;
    int32_t n_embd         = 0;
    int32_t layer_end      = 0;
    int32_t pos_start      = 0;
    size_t payload_bytes   = 0;
    size_t header_bytes    = 0;
    size_t alignment       = alignof(float);
    size_t stride          = sizeof(float);
    const char * dtype     = "f32";
    double latency_ms      = 0.0;
    std::string sha256;
    bool memcmp_ok         = true;
    std::string dump_path_before;
    std::string dump_path_after;
};

bool dist_debug_transport_dump_enabled();

bool hidden_write_bin(const std::string & path, const float * data, size_t nfloats);

bool hidden_compare_bins(
        const std::string & path_a,
        const std::string & path_b,
        size_t * diff_offset = nullptr);

bool hidden_memcmp_buffers(
        const float * a,
        const float * b,
        size_t nfloats,
        size_t * diff_offset = nullptr);

std::string hidden_transport_trace_json(const hidden_transport_trace & tr);

// Dump pre-TCP hidden and verify post-TCP recv via memcmp.
hidden_transport_trace dist_debug_transport_send(
        int32_t step,
        const char * phase,
        const char * link,
        int32_t n_tokens,
        int32_t n_embd,
        int32_t layer_end,
        int32_t pos_start,
        const float * data,
        double latency_ms = 0.0);

hidden_transport_trace dist_debug_transport_recv(
        int32_t step,
        const char * phase,
        const char * link,
        int32_t n_tokens,
        int32_t n_embd,
        int32_t layer_end,
        int32_t pos_start,
        const float * data,
        double latency_ms = 0.0);

// Verify llama_set_hidden_state roundtrip inside context.
struct hidden_state_api_check {
    bool ok = false;
    size_t diff_offset = SIZE_MAX;
    std::string message;
};

hidden_state_api_check verify_hidden_state_roundtrip(
        llama_context * ctx,
        const float * data,
        int32_t n_tokens,
        int32_t n_embd);
