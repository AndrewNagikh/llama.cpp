#pragma once

#include "architecture/semantic_runtime_descriptor.h"
#include "layer_store.h"
#include "manifest_builder/manifest_builder.h"

#include <cstdint>
#include <string>

// Task 9.9 — descriptor-driven partial GGUF assembly (no include_embedding flags).

bool materialize_worker_gguf(
        const layer_store & store,
        const model_manifest & manifest,
        const semantic_runtime_descriptor & rt,
        worker_role role,
        int32_t layer_start,
        int32_t layer_end,
        const std::string & output_path,
        std::string & err);
