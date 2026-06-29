#pragma once

#include "llama.h"

#include <string>

struct runtime_api_verify_result {
    bool ok = false;
    size_t diff_offset = 0;
    std::string message;
};

runtime_api_verify_result verify_runtime_hidden_api(
        const std::string & model_path,
        int32_t n_tokens = 1);
