#pragma once

#include "orchestrator/manifest_builder/manifest_builder.h"

#include <cstdint>
#include <string>

enum class tensor_semantic_role {
    unknown,
    embedding,
    output_head,
    output_norm,
    input_norm,
    transformer_layer,
    attention,
    ffn,
    expert,
    router,
    shared_expert,
    rotary,
    bias,
    gate,
    metadata,
    other,
};

std::string tensor_semantic_role_to_string(tensor_semantic_role role);
tensor_semantic_role tensor_semantic_role_from_string(const std::string & s);

void classify_tensor_enhanced(const std::string & name, int32_t & layer, tensor_role & role);
tensor_semantic_role classify_tensor_semantic(
        const std::string & name,
        tensor_role manifest_role,
        int32_t layer);
