#pragma once

#include "manifest_builder/manifest_builder.h"
#include "verification_types.h"

#include <string>
#include <vector>

struct manifest_verify_result {
    verify_check_result summary;
    std::vector<std::string> issues;
};

manifest_verify_result manifest_verify_against_gguf(
        const model_manifest & manifest,
        const std::string & gguf_path);
