#pragma once

#include "layer_store.h"
#include "layer_special.h"
#include "manifest_builder/manifest_builder.h"

#include <cstdint>
#include <string>

// Fetch and cache GGUF metadata region [0, tensor_data_offset) for assembly.
bool layer_store_cache_metadata(
        layer_store & store,
        const model_manifest & manifest,
        const std::string & source_url);

// Build a loadable GGUF from cached metadata + layer blobs at original offsets.
bool layer_store_materialize_gguf(
        const layer_store & store,
        const model_manifest & manifest,
        const std::string & output_path,
        int32_t layer_start,
        int32_t layer_end,
        bool include_embedding,
        bool include_output);

// Metadata-only GGUF shell for orchestrator tokenization (no weight blobs required).
bool layer_store_materialize_tokenizer_shell(
        const layer_store & store,
        const model_manifest & manifest,
        const std::string & output_path);
