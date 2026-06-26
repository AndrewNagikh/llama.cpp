#include "architecture/plugins/llama_plugin.h"

#include "architecture/blob_builder.h"

#include <algorithm>

namespace {

bool arch_is_moe(const std::string & arch) {
    return arch.find("moe") != std::string::npos;
}

void build_entry_globals(architecture_descriptor & desc, const model_manifest & manifest) {
    for (const auto & tensor : manifest.tensors) {
        if (tensor.layer >= 0) {
            continue;
        }

        const tensor_semantic_role semantic =
                classify_tensor_semantic(tensor.name, tensor.role, tensor.layer);
        if (semantic == tensor_semantic_role::output_head ||
                semantic == tensor_semantic_role::output_norm) {
            continue;
        }

        std::string blob_id = "globals";
        blob_deploy_target deploy = blob_deploy_target::entry_node;
        if (semantic == tensor_semantic_role::embedding) {
            blob_id = "embedding";
        } else if (semantic == tensor_semantic_role::input_norm) {
            blob_id = "input_norm";
        } else if (semantic == tensor_semantic_role::rotary) {
            blob_id = "rope";
        }

        semantic_blob * blob = ensure_blob(desc, blob_id, semantic, deploy);
        add_tensor_to_blob(*blob, tensor);
    }
}

void build_output_blobs(
        architecture_descriptor & desc,
        const model_manifest & manifest,
        const bool force_separate_head) {
    if (const tensor_descriptor * norm = manifest_find_role(manifest, tensor_role::output_norm)) {
        semantic_blob * blob = ensure_blob(
                desc,
                "output_norm",
                tensor_semantic_role::output_norm,
                blob_deploy_target::final_node);
        add_tensor_to_blob(*blob, *norm);
    }

    const bool has_lm_head = manifest_has_tensor(manifest, "output.weight") || force_separate_head;
    if (!has_lm_head) {
        desc.tied_embeddings  = true;
        desc.separate_lm_head = false;

        semantic_blob * head = ensure_blob(
                desc,
                "output_head",
                tensor_semantic_role::output_head,
                blob_deploy_target::final_node);
        head->storage_alias     = true;
        head->storage_blob_id   = "embedding";

        if (const tensor_descriptor * emb = manifest_find_role(manifest, tensor_role::embedding)) {
            add_tensor_to_blob(*head, *emb);
        }

        if (semantic_blob * embedding = find_blob_mut(desc.blobs, "embedding")) {
            embedding->deploy = blob_deploy_target::all_nodes;
        }
        return;
    }

    desc.tied_embeddings  = false;
    desc.separate_lm_head = true;

    if (const tensor_descriptor * head_tensor = manifest_find_role(manifest, tensor_role::lm_head)) {
        semantic_blob * blob = ensure_blob(
                desc,
                "output_head",
                tensor_semantic_role::output_head,
                blob_deploy_target::final_node);
        add_tensor_to_blob(*blob, *head_tensor);
    }
}

} // namespace

std::string llama_architecture_plugin::family() const {
    return "llama";
}

bool llama_architecture_plugin::matches(const model_manifest & manifest) const {
    const std::string & arch = manifest.architecture;
    if (arch_prefix_matches(arch, "qwen")) {
        return false;
    }
    if (arch_prefix_matches(arch, "gemma")) {
        return false;
    }
    return arch_prefix_matches(arch, "llama") ||
            arch_prefix_matches(arch, "tinyllama") ||
            arch_prefix_matches(arch, "mistral") ||
            arch_prefix_matches(arch, "phi") ||
            arch_prefix_matches(arch, "deepseek") ||
            arch_prefix_matches(arch, "smollm");
}

architecture_descriptor llama_architecture_plugin::build_descriptor(
        const model_manifest & manifest) const {
    architecture_descriptor desc;
    desc.architecture = manifest.architecture;
    desc.family       = family();
    desc.is_moe       = arch_is_moe(manifest.architecture);

    build_entry_globals(desc, manifest);
    build_output_blobs(desc, manifest, false);
    populate_layer_blobs(desc, manifest);
    set_dense_worker_requirements(desc);
    return desc;
}
