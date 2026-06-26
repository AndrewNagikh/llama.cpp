#pragma once

#include "verification_types.h"

#include <string>
#include <vector>

struct tensor_verify_entry {
    std::string name;
    uint64_t    offset = 0;
    uint64_t    size_bytes = 0;
    std::string ggml_type;
    std::string sha256_original;
    std::string sha256_materialized;
    bool        match = false;
};

struct tensor_verify_result {
    verify_check_result summary;
    std::vector<tensor_verify_entry> tensors;
    std::vector<std::string> mismatches;
};

tensor_verify_result tensor_verify_files(
        const std::string & original,
        const std::string & materialized);
