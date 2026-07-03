#include "distributed_runtime_descriptor.h"

#include "architecture/semantic_roles.h"

#include "architecture/semantic_blob.h"

namespace {

std::string role_to_string(worker_role role) {
    switch (role) {
        case worker_role::entry:  return "ENTRY";
        case worker_role::middle: return "MIDDLE";
        case worker_role::final:  return "FINAL";
        case worker_role::full:   return "FULL";
        case worker_role::tokenizer:    return "TOKENIZER";
        case worker_role::embedding:    return "EMBEDDING";
        case worker_role::pipeline_stage: return "PIPELINE_STAGE";
        case worker_role::output_head: return "OUTPUT_HEAD";
        case worker_role::sampler:    return "SAMPLER";
    }
    return "UNKNOWN";
}

std::string blob_role_to_string(tensor_semantic_role role) {
    return tensor_semantic_role_to_string(role);
}

} // namespace

nlohmann::json distributed_runtime_descriptor::to_json() const {
    nlohmann::json blobs = nlohmann::json::array();
    for (const semantic_blob & blob : semantic.blobs) {
        nlohmann::json stages = nlohmann::json::object();
        const auto it = blob_stages.find(blob.id);
        if (it != blob_stages.end()) {
            stages = {
                { "entry",  it->second.required_for_entry },
                { "middle", it->second.required_for_middle },
                { "final",  it->second.required_for_final },
            };
        }
        blobs.push_back({
            { "id", blob.id },
            { "type", blob_role_to_string(blob.role) },
            { "tensor_count", blob.tensors.size() },
            { "storage_alias", blob.storage_alias },
            { "stages", stages },
        });
    }

    nlohmann::json workers = nlohmann::json::array();
    for (const worker_descriptor & w : semantic.workers) {
        workers.push_back({
            { "role", role_to_string(w.role) },
            { "required_blobs", w.required_blob_ids },
        });
    }

    return {
        { "architecture", semantic.architecture },
        { "family", semantic.family },
        { "tied_embeddings", semantic.tied_embeddings },
        { "separate_lm_head", semantic.separate_lm_head },
        { "is_moe", semantic.is_moe },
        { "capabilities", {
            { "partial_forward", capabilities.supports_partial_forward },
            { "hidden_injection", capabilities.supports_hidden_injection },
            { "embedding_injection", capabilities.supports_embedding_injection },
        }},
        { "semantic_blobs", blobs },
        { "runtime_requirements", workers },
    };
}

distributed_runtime_descriptor build_distributed_runtime_descriptor(
        const model_manifest & manifest) {
    distributed_runtime_descriptor desc;
    desc.semantic = build_semantic_runtime_descriptor(manifest);
    desc.blob_stages = compute_blob_stage_requirements(desc.semantic);
    return desc;
}
