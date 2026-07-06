#pragma once

#include "architecture/semantic_runtime_descriptor.h"
#include "architecture/worker_requirement.h"
#include "node_agent/layer_store/layer_store.h"
#include "orchestrator/manifest_builder/manifest_builder.h"

#include <string>

// Task 11.8 — compatibility-only GGUF assembly (verification, export, parity tests).
// Not used on the layer-first inference hot path when DIST_RUNTIME_LAYER_FIRST=1.

bool runtime_compat_materialize_worker_gguf(
        const layer_store & store,
        const model_manifest & manifest,
        const semantic_runtime_descriptor & rt,
        worker_role role,
        int32_t layer_start,
        int32_t layer_end,
        const std::string & output_path,
        std::string & err);
