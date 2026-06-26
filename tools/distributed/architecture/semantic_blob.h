#pragma once

#include "semantic_roles.h"

#include <cstdint>
#include <string>
#include <vector>

struct semantic_tensor_slot {
    std::string name;
    uint64_t    offset     = 0;
    uint64_t    size_bytes = 0;
};

enum class blob_deploy_target {
    entry_node,
    final_node,
    all_nodes,
    none,
};

struct semantic_blob {
    std::string           id;
    tensor_semantic_role  role = tensor_semantic_role::unknown;
    std::vector<semantic_tensor_slot> tensors;
    blob_deploy_target    deploy = blob_deploy_target::none;
    bool                  storage_alias = false;
    std::string           storage_blob_id;
};

const semantic_blob * find_blob(const std::vector<semantic_blob> & blobs, const std::string & id);
semantic_blob * find_blob_mut(std::vector<semantic_blob> & blobs, const std::string & id);

std::string blob_checksum_key(const std::string & blob_id, const std::string & tensor_name);
std::string layer_blob_id(int32_t layer_index);
