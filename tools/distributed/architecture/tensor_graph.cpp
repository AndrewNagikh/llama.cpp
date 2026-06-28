#include "tensor_graph.h"

#include "blob_builder.h"
#include "semantic_roles.h"

namespace {

std::string global_blob_id(const tensor_semantic_role semantic) {
    switch (semantic) {
        case tensor_semantic_role::embedding:   return "embedding";
        case tensor_semantic_role::output_head: return "output_head";
        case tensor_semantic_role::output_norm: return "output_norm";
        case tensor_semantic_role::input_norm:  return "input_norm";
        case tensor_semantic_role::rotary:      return "rope";
        default:                                return "globals";
    }
}

blob_deploy_target deploy_for_global(const tensor_semantic_role semantic) {
    switch (semantic) {
        case tensor_semantic_role::embedding:
        case tensor_semantic_role::input_norm:
            return blob_deploy_target::entry_node;
        case tensor_semantic_role::output_head:
        case tensor_semantic_role::output_norm:
            return blob_deploy_target::final_node;
        case tensor_semantic_role::rotary:
            return blob_deploy_target::all_nodes;
        default:
            return blob_deploy_target::entry_node;
    }
}

bool arch_is_moe(const std::string & arch) {
    return arch.find("moe") != std::string::npos;
}

model_manifest manifest_view_from_graph(const tensor_graph & graph) {
    model_manifest view;
    view.architecture = graph.architecture;
    view.n_layer      = 0;
    for (const auto & entry : graph.tensors) {
        view.tensors.push_back(entry.tensor);
        if (entry.layer_index >= 0) {
            view.n_layer = std::max(view.n_layer,
                    static_cast<uint32_t>(entry.layer_index + 1));
        }
    }
    for (size_t i = 0; i < view.n_layer; ++i) {
        layer_descriptor ld;
        ld.layer_index = static_cast<int32_t>(i);
        view.layers.push_back(ld);
    }
    return view;
}

} // namespace

tensor_graph analyze_tensor_graph(const model_manifest & manifest) {
    tensor_graph graph;
    graph.architecture = manifest.architecture;
    graph.tensors.reserve(manifest.tensors.size());

    for (const auto & tensor : manifest.tensors) {
        classified_tensor entry;
        entry.tensor        = tensor;
        entry.layer_index   = tensor.layer;
        entry.semantic      = classify_tensor_semantic(
                tensor.name, tensor.role, tensor.layer);
        graph.tensors.push_back(std::move(entry));
    }
    return graph;
}

architecture_descriptor build_descriptor_from_graph(const tensor_graph & graph) {
    architecture_descriptor desc;
    const model_manifest view = manifest_view_from_graph(graph);

    desc.architecture     = graph.architecture;
    desc.is_moe           = arch_is_moe(graph.architecture);
    desc.separate_lm_head = manifest_has_tensor(view, "output.weight");
    desc.tied_embeddings  = !desc.separate_lm_head;

    for (const auto & entry : graph.tensors) {
        if (entry.layer_index >= 0) {
            continue;
        }
        if (entry.semantic == tensor_semantic_role::output_head ||
                entry.semantic == tensor_semantic_role::output_norm) {
            continue;
        }

        const std::string blob_id = global_blob_id(entry.semantic);
        semantic_blob * blob = ensure_blob(
                desc,
                blob_id,
                entry.semantic,
                deploy_for_global(entry.semantic));
        add_tensor_to_blob(*blob, entry.tensor);
    }

    if (const tensor_descriptor * norm = manifest_find_role(view, tensor_role::output_norm)) {
        semantic_blob * blob = ensure_blob(
                desc,
                "output_norm",
                tensor_semantic_role::output_norm,
                blob_deploy_target::final_node);
        add_tensor_to_blob(*blob, *norm);
    }

    if (desc.separate_lm_head) {
        if (const tensor_descriptor * head = manifest_find_role(view, tensor_role::lm_head)) {
            semantic_blob * blob = ensure_blob(
                    desc,
                    "output_head",
                    tensor_semantic_role::output_head,
                    blob_deploy_target::final_node);
            add_tensor_to_blob(*blob, *head);
        }
    } else {
        semantic_blob * head = ensure_blob(
                desc,
                "output_head",
                tensor_semantic_role::output_head,
                blob_deploy_target::final_node);
        head->storage_alias   = true;
        head->storage_blob_id = "embedding";
        if (const tensor_descriptor * emb = manifest_find_role(view, tensor_role::embedding)) {
            add_tensor_to_blob(*head, *emb);
        }
        if (semantic_blob * embedding = find_blob_mut(desc.blobs, "embedding")) {
            embedding->deploy = blob_deploy_target::all_nodes;
        }
    }

    populate_layer_blobs(desc, view);
    set_dense_worker_requirements(desc);
    return desc;
}
