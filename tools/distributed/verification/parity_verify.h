#pragma once

#include "verification_types.h"

#include <string>
#include <vector>

struct parity_result {
    verify_check_result logits;
    verify_check_result sampling;
    std::vector<int32_t> tokens_original;
    std::vector<int32_t> tokens_materialized;
};

parity_result verify_inference_parity(
        const std::string & original_gguf,
        const std::string & materialized_gguf,
        const std::string & prompt = "The capital of France is",
        int max_tokens = 32);
