#pragma once

#include "layer_store/layer_store.h"
#include "manifest_builder/manifest_builder.h"

#include <string>

bool verify_worker_materialization(
        const layer_store & store,
        const model_manifest & manifest,
        int32_t layer_start,
        int32_t layer_end,
        bool include_embedding,
        bool include_output,
        std::string & err);
