#pragma once

#include "verification_types.h"

#include <string>

struct load_verify_result {
    verify_check_result summary;
    int32_t n_layer      = 0;
    int32_t n_gpu_layers = 0;
    int64_t tensor_count = 0;
    int32_t kv_size      = 0;
    std::string metadata_sha256;
};

load_verify_result load_verify_gguf(const std::string & gguf_path);
