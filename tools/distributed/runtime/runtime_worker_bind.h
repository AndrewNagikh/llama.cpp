#pragma once

#include "architecture/worker_requirement.h"
#include "node_agent/layer_store/layer_store.h"
#include "orchestrator/manifest_builder/manifest_builder.h"

#include <cstdint>
#include <string>

// Task 11.6/11.7 — verify Layer Store tensors and resolve cached worker GGUF without assembly.

struct runtime_bind_result {
    bool        success            = false;
    bool        tensors_ready        = false;
    bool        cached_gguf_ready    = false;
    bool        materialize_required = true;
    std::string worker_gguf_path;
    std::string error;
};

runtime_bind_result runtime_bind_worker(
        const layer_store & store,
        const model_manifest & manifest,
        worker_role role,
        int32_t layer_start,
        int32_t layer_end);

bool runtime_verify_worker_tensors(
        const layer_store & store,
        const model_manifest & manifest,
        worker_role role,
        int32_t layer_start,
        int32_t layer_end,
        std::string & err);
