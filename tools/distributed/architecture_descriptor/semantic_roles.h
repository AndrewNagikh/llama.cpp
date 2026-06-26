#pragma once

#include <cstdint>
#include <string>

// Universal semantic roles for GGUF tensors.  These are coarser than per-tensor
// manifest roles and drive worker tensor plans without model-family conditionals.

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

enum class worker_deploy_role {
    entry,
    middle,
    final,
    full,
};

struct tensor_role_requirement {
    tensor_semantic_role semantic = tensor_semantic_role::unknown;
    bool required_for_entry       = false;
    bool required_for_middle      = false;
    bool required_for_final       = false;
    bool replicate_to_all_nodes   = false;
};

std::string tensor_semantic_role_to_string(tensor_semantic_role role);
tensor_semantic_role tensor_semantic_role_from_string(const std::string & s);

std::string worker_deploy_role_to_string(worker_deploy_role role);
