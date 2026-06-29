#pragma once

#include "runtime_debug/trace_recorder.h"

#include "llama.h"

#include <string>
#include <vector>

struct mono_decode_step {
    int32_t step           = 0;
    std::string phase;
    int32_t input_token    = -1;
    int32_t position       = 0;
    tensor_stats hidden_stats{};
    tensor_stats logits_stats{};
    int32_t selected_token = -1;
    int32_t argmax_token   = -1;
    float argmax_score     = 0.0f;
    int32_t kv_entries     = 0;
    int32_t seq_len        = 0;
    std::vector<float> hidden;
    std::vector<float> logits;
};

struct mono_trace_result {
    std::string trace_path;
    std::vector<mono_decode_step> steps;
    std::vector<int32_t> prompt_tokens;
    bool ok = false;
    std::string error;
};

mono_trace_result run_monolithic_trace(
        const std::string & model_path,
        const std::string & prompt,
        int max_tokens,
        const std::string & session_id = "mono");

mono_trace_result load_trace_file(const std::string & jsonl_path);
