#pragma once

#include "layer_store/layer_store.h"
#include "manifest_builder/manifest_builder.h"
#include "architecture/worker_requirement.h"

#include <string>

// Descriptor-driven worker materialization verification (Task 9.9).

bool verify_worker_materialization_for_role(
        const layer_store & store,
        const model_manifest & manifest,
        worker_role role,
        int32_t layer_start,
        int32_t layer_end,
        std::string & err);

// Legacy flag API — delegates to verify_worker_materialization_for_role.
bool verify_worker_materialization(
        const layer_store & store,
        const model_manifest & manifest,
        int32_t layer_start,
        int32_t layer_end,
        bool include_embedding,
        bool include_output,
        std::string & err);
