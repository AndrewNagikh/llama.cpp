#pragma once

#include "verification_types.h"

#include "nlohmann/json.hpp"

#include <string>
#include <vector>

struct gguf_diff_result {
    verify_check_result header;
    verify_check_result metadata;
    verify_check_result tensor_directory;
    std::vector<std::string> differences;
    bool passed = false;

    nlohmann::json to_json() const;
};

gguf_diff_result gguf_diff_files(const std::string & original, const std::string & materialized);
