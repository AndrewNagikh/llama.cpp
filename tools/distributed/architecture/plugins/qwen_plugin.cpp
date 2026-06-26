#include "architecture/plugins/qwen_plugin.h"

#include "architecture/blob_builder.h"

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
        if (semantic == tensor_semantic_role::embedding) {
            blob_id = "embedding";
        } else if (semantic == tensor_semantic_role::input_norm) {
            blob_id = "input_norm";
        } else if (semantic == tensor_semantic_role::rotary) {
            blob_id = "rope";
        }

        semantic_blob * blob = ensure_blob(
                desc,
                blob_id,
                semantic,
                blob_deploy_target::entry_node);
        add_tensor_to_blob(*blob, tensor);
    }
}

} // namespace

std::string qwen_architecture_plugin::family() const {
    return "qwen";
}

bool qwen_architecture_plugin::matches(const model_manifest & manifest) const {
    return arch_prefix_matches(manifest.architecture, "qwen");
}

architecture_descriptor qwen_architecture_plugin::build_descriptor(
        const model_manifest & manifest) const {
    architecture_descriptor desc;
    desc.architecture = manifest.architecture;
    desc.family       = family();
    desc.is_moe       = arch_is_moe(manifest.architecture);

    build_entry_globals(desc, manifest);

    if (const tensor_descriptor * norm = manifest_find_role(manifest, tensor_role::output_norm)) {
        semantic_blob * blob = ensure_blob(
                desc,
                "output_norm",
                tensor_semantic_role::output_norm,
                blob_deploy_target::final_node);
        add_tensor_to_blob(*blob, *norm);
    }

    desc.separate_lm_head = manifest_has_tensor(manifest, "output.weight");
    desc.tied_embeddings  = !desc.separate_lm_head;

    if (desc.separate_lm_head) {
        if (const tensor_descriptor * head = manifest_find_role(manifest, tensor_role::lm_head)) {
            semantic_blob * blob = ensure_blob(
                    desc,
                    "output_head",
                    tensor_semantic_role::output_head,
                    blob_deploy_target::final_node);
            add_tensor_to_blob(*blob, *head);
        }
    } else if (const tensor_descriptor * emb = manifest_find_role(manifest, tensor_role::embedding)) {
        semantic_blob * head = ensure_blob(
                desc,
                "output_head",
                tensor_semantic_role::output_head,
                blob_deploy_target::final_node);
        head->storage_alias   = true;
        head->storage_blob_id = "embedding";
        add_tensor_to_blob(*head, *emb);
        if (semantic_blob * embedding = find_blob_mut(desc.blobs, "embedding")) {
            embedding->deploy = blob_deploy_target::all_nodes;
        }
    }

    populate_layer_blobs(desc, manifest);
    set_dense_worker_requirements(desc);
    return desc;
}
