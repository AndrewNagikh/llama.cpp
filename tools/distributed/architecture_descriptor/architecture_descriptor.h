#pragma once

#include "semantic_roles.h"

#include "orchestrator/manifest_builder/manifest_builder.h"

#include <map>
#include <string>
#include <vector>

struct semantic_tensor_entry {
    std::string           name;
    tensor_semantic_role  semantic = tensor_semantic_role::unknown;
    tensor_role           manifest_role = tensor_role::unknown;
    int32_t               layer = -1;
};

struct architecture_descriptor {
    std::string architecture;
    uint32_t    n_layer = 0;

    bool tied_embeddings       = false;
    bool has_separate_output   = false;
    bool has_output_norm       = false;
    bool is_moe                = false;

    std::map<std::string, std::string> special_tensors;
    std::vector<semantic_tensor_entry>  tensors;
    std::vector<tensor_role_requirement> role_requirements;

    bool empty() const { return architecture.empty() && tensors.empty(); }
};

architecture_descriptor build_architecture_descriptor(const model_manifest & manifest);

void classify_tensor_enhanced(const std::string & name, int32_t & layer, tensor_role & role);
tensor_semantic_role classify_tensor_semantic(
        const std::string & name,
        tensor_role manifest_role,
        int32_t layer);

bool architecture_worker_needs_embedding_blob(
        const architecture_descriptor & desc,
        worker_deploy_role role);

bool architecture_should_replicate_embedding(const architecture_descriptor & desc);

bool architecture_materialize_needs_embedding_for_output(
        const architecture_descriptor & desc,
        bool include_embedding,
        bool include_output);

bool architecture_tensor_included(
        const architecture_descriptor & desc,
        const tensor_descriptor & tensor,
        int32_t layer_start,
        int32_t layer_end,
        bool include_embedding,
        bool include_output);

bool architecture_output_satisfied_by_embedding(const architecture_descriptor & desc);
