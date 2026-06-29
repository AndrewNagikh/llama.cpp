#pragma once

#include "decode_loop_parity.h"

#include <string>
#include <vector>

struct decode_state_row {
    int32_t step = -1;
    std::string phase;
    std::string worker;
    int32_t token = -1;
    int32_t prev_token = -1;
    int32_t position = -1;
    int32_t n_past = 0;
    int32_t kv_entries = 0;
    std::string hidden_sha256;
    bool has_runtime_state = false;
};

struct decode_state_parity_report {
    bool all_pass = false;
    int32_t first_fail_step = -1;
    std::string first_fail_worker;
    std::string field;
    std::string message;
    std::vector<decode_state_row> rows;
};

decode_state_parity_report compare_decode_state_traces(
        const parsed_trace & mono,
        const parsed_trace & entry,
        const parsed_trace & middle,
        const parsed_trace & final);

decode_state_parity_report compare_runtime_state_vs_mono(
        const std::string & mono_trace_path,
        const std::string & entry_trace_path,
        const std::string & middle_trace_path,
        const std::string & final_trace_path);
