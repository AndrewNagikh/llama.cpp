#include "tensor_plan.h"

#include "worker_requirement.h"

#include <algorithm>

namespace {

worker_role role_from_flags(const bool include_embedding, const bool include_output) {
    if (include_embedding && include_output) {
        return worker_role::full;
    }
    if (include_embedding) {
        return worker_role::entry;
    }
    if (include_output) {
        return worker_role::final;
    }
    return worker_role::middle;
}

bool tensor_in_blob(const semantic_blob & blob, const std::string & name) {
    for (const auto & slot : blob.tensors) {
        if (slot.name == name) {
            return true;
        }
    }
    return false;
}

} // namespace

bool descriptor_tensor_included(
        const architecture_descriptor & desc,
        const tensor_descriptor & tensor,
        const int32_t layer_start,
        const int32_t layer_end,
        const bool include_embedding,
        const bool include_output) {
    if (tensor.layer >= layer_start && tensor.layer < layer_end) {
        return true;
    }

    const worker_role role = role_from_flags(include_embedding, include_output);
    const auto plan = materialize_plan_for_worker(
            desc.worker_requirements, role, layer_start, layer_end);

    for (const std::string & blob_id : plan.required_blobs) {
        const semantic_blob * blob = find_blob(desc.blobs, blob_id);
        if (blob != nullptr && tensor_in_blob(*blob, tensor.name)) {
            return true;
        }
    }

    if (include_embedding) {
        for (const semantic_blob & blob : desc.blobs) {
            if (blob.deploy == blob_deploy_target::entry_node ||
                    blob.deploy == blob_deploy_target::all_nodes) {
                if (tensor_in_blob(blob, tensor.name)) {
                    return true;
                }
            }
        }
    }

    if (include_output) {
        for (const semantic_blob & blob : desc.blobs) {
            if (blob.deploy == blob_deploy_target::final_node ||
                    blob.role == tensor_semantic_role::output_head ||
                    blob.role == tensor_semantic_role::output_norm) {
                if (tensor_in_blob(blob, tensor.name)) {
                    return true;
                }
            }
        }
    }

    return false;
}
