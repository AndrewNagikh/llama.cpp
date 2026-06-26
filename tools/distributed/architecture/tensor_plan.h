#pragma once

#include "architecture_descriptor.h"
#include "orchestrator/manifest_builder/manifest_builder.h"

bool descriptor_tensor_included(
        const architecture_descriptor & desc,
        const tensor_descriptor & tensor,
        int32_t layer_start,
        int32_t layer_end,
        bool include_embedding,
        bool include_output);
