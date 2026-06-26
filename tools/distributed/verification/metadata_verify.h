#pragma once

#include "verification_types.h"

#include <string>
#include <vector>

struct metadata_verify_result {
    verify_check_result summary;
    std::vector<std::string> differences;
};

metadata_verify_result metadata_verify_kv_prefixes(
        const std::string & original,
        const std::string & materialized,
        const std::vector<std::string> & prefixes = {
            "tokenizer.",
            "general.",
            "rope.",
            "chat_template.",
            "special_vocab.",
            "quantization.",
        });
