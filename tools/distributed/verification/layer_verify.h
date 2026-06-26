#pragma once

#include "layer_store/layer_store.h"
#include "manifest_builder/manifest_builder.h"
#include "verification_types.h"

#include <string>

struct layer_verify_result {
    verify_check_result summary;
    std::vector<std::string> issues;
};

layer_verify_result layer_verify_store(
        const layer_store & store,
        const model_manifest & manifest,
        const std::string & source_path);
