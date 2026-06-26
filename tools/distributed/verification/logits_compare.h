#pragma once

#include "verification_types.h"

#include <string>
#include <vector>

struct logits_compare_stats {
    size_t n_elements    = 0;
    float  max_value     = 0.0f;
    int32_t max_index    = -1;
    float  mean_abs_err  = 0.0f;
    float  max_abs_err   = 0.0f;
    bool   match         = false;
};

struct logits_compare_result {
    verify_check_result summary;
    logits_compare_stats original;
    logits_compare_stats materialized;
    std::vector<float> logits_original;
    std::vector<float> logits_materialized;
};

logits_compare_result compare_logits_files(
        const std::string & original_gguf,
        const std::string & materialized_gguf,
        const std::string & prompt = "The capital of France is");
