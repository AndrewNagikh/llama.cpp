#pragma once

#include "runtime_debug/hidden_transport.h"

#include <string>

struct hidden_transport_verify_result {
    bool ok = false;
    bool tcp_memcmp_ok = true;
    bool api_roundtrip_ok = true;
    size_t tcp_diff_offset = 0;
    size_t api_diff_offset = 0;
    std::string message;
    hidden_transport_trace send_trace{};
    hidden_transport_trace recv_trace{};
};

// Loopback: send hidden over TCP socket pair and memcmp.
hidden_transport_verify_result verify_hidden_tcp_loopback(
        int32_t n_tokens,
        int32_t n_embd,
        const float * data);

// Compare worker output bin vs received bin from transport trace dir.
hidden_transport_verify_result verify_hidden_transport_files(
        const std::string & before_path,
        const std::string & after_path);
