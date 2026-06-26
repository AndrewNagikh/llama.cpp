#pragma once

#include "layer_store/layer_store.h"
#include "manifest_builder/manifest_builder.h"
#include "verification_types.h"

#include <string>

struct materialization_verify_result {
    verify_check_result summary;
    std::string sha256_original;
    std::string sha256_materialized;
    std::string sha256_repeat;
};

materialization_verify_result verify_materialization_repeatability(
        const layer_store & store,
        const model_manifest & manifest,
        const std::string & output_dir);

bool materialize_full_model(
        const layer_store & store,
        const model_manifest & manifest,
        const std::string & output_path);
