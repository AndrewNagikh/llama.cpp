#include "blob_builder.h"

#include <algorithm>

bool arch_prefix_matches(const std::string & architecture, const char * prefix) {
    return architecture.rfind(prefix, 0) == 0;
}

const tensor_descriptor * manifest_find_tensor(
        const model_manifest & manifest,
        const std::string & name) {
    for (const auto & tensor : manifest.tensors) {
        if (tensor.name == name) {
            return &tensor;
        }
    }
    return nullptr;
}

const tensor_descriptor * manifest_find_role(
        const model_manifest & manifest,
        const tensor_role role) {
    for (const auto & tensor : manifest.tensors) {
        if (tensor.role == role) {
            return &tensor;
        }
    }
    return nullptr;
}

bool manifest_has_tensor(const model_manifest & manifest, const std::string & name) {
    return manifest_find_tensor(manifest, name) != nullptr;
}

semantic_blob * ensure_blob(
        architecture_descriptor & desc,
        const std::string & id,
        const tensor_semantic_role role,
        const blob_deploy_target deploy) {
    if (auto * existing = find_blob_mut(desc.blobs, id)) {
        return existing;
    }
    desc.blobs.push_back({ id, role, {}, deploy, false, "" });
    return &desc.blobs.back();
}

void add_tensor_to_blob(semantic_blob & blob, const tensor_descriptor & tensor) {
    for (const auto & slot : blob.tensors) {
        if (slot.name == tensor.name) {
            return;
        }
    }
    semantic_tensor_slot slot;
    slot.name       = tensor.name;
    slot.offset     = tensor.offset;
    slot.size_bytes = tensor.size_bytes;
    blob.tensors.push_back(std::move(slot));
}

void populate_layer_blobs(architecture_descriptor & desc, const model_manifest & manifest) {
    for (const auto & ld : manifest.layers) {
        const std::string id = layer_blob_id(ld.layer_index);
        auto * blob = ensure_blob(
                desc,
                id,
                tensor_semantic_role::transformer_layer,
                blob_deploy_target::none);
        for (const auto & tensor : manifest.tensors) {
            if (tensor.layer == ld.layer_index) {
                add_tensor_to_blob(*blob, tensor);
            }
        }
    }
}

void set_dense_worker_requirements(architecture_descriptor & desc) {
    desc.worker_requirements.clear();

    worker_requirement entry;
    entry.role = worker_role::entry;
    entry.required_blobs = { "embedding" };
    if (find_blob(desc.blobs, "rope")) {
        entry.required_blobs.push_back("rope");
    }
    if (find_blob(desc.blobs, "input_norm")) {
        entry.required_blobs.push_back("input_norm");
    }
    if (find_blob(desc.blobs, "globals")) {
        entry.required_blobs.push_back("globals");
    }
    desc.worker_requirements.push_back(std::move(entry));

    worker_requirement middle;
    middle.role = worker_role::middle;
    desc.worker_requirements.push_back(std::move(middle));

    worker_requirement final_req;
    final_req.role = worker_role::final;
    final_req.required_blobs = { "output_norm", "output_head" };
    desc.worker_requirements.push_back(std::move(final_req));

    worker_requirement full;
    full.role = worker_role::full;
    full.required_blobs = { "embedding", "output_norm", "output_head" };
    if (find_blob(desc.blobs, "rope")) {
        full.required_blobs.push_back("rope");
    }
    desc.worker_requirements.push_back(std::move(full));
}
