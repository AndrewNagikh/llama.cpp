#pragma once

#include "verification_types.h"

#include <string>
#include <vector>

struct hidden_state_stats {
    size_t   n_elements   = 0;
    std::string sha256;
    float    mean_abs     = 0.0f;
    float    max_abs      = 0.0f;
    float    mae_vs_other = 0.0f;
    float    max_err      = 0.0f;
    bool     match        = false;
};

struct hidden_state_compare_result {
    verify_check_result summary;
    hidden_state_stats original;
    hidden_state_stats materialized;
};

hidden_state_compare_result compare_hidden_states(
        const std::string & original_gguf,
        const std::string & materialized_gguf,
        const std::string & prompt = "The capital of France is");
