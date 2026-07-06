#pragma once

#include "decode_loop_parity.h"

#include <string>

struct hidden_pipeline_parity_result {
    bool ok = false;
    parity_metrics metrics{};
    tensor_stats reference_stats{};
    tensor_stats producer_stats{};
    bool exact_sha_match = false;
    bool aggregate_match = false;
    std::string message;
};

hidden_pipeline_parity_result verify_hidden_at_layer_boundary(
        const std::string & model_path,
        const std::string & prompt,
        int32_t layer_end,
        const parsed_trace & entry_trace);
