#include "architecture_plugin.h"

#include "architecture/tensor_graph.h"

semantic_runtime_descriptor architecture_plugin::build_runtime_descriptor(
        const model_manifest & manifest) const {
    semantic_runtime_descriptor rt = build_semantic_runtime_descriptor(manifest);
    if (rt.family.empty()) {
        rt.family = family();
    }
    return rt;
}

distributed_runtime_descriptor architecture_plugin::build_distributed_descriptor(
        const model_manifest & manifest) const {
    distributed_runtime_descriptor desc;
    desc.semantic = build_runtime_descriptor(manifest);
    desc.blob_stages = compute_blob_stage_requirements(desc.semantic);
    return desc;
}

bool architecture_plugin::verify_runtime(
        const distributed_runtime_descriptor & desc,
        std::string & error) const {
    if (desc.empty()) {
        error = "empty runtime descriptor";
        return false;
    }
    if (desc.semantic.blobs.empty()) {
        error = "no semantic blobs";
        return false;
    }
    if (desc.semantic.workers.empty()) {
        error = "no runtime worker requirements";
        return false;
    }
    const semantic_blob * embedding = find_blob(desc.semantic.blobs, "embedding");
    if (embedding == nullptr || embedding->tensors.empty()) {
        error = "missing embedding blob";
        return false;
    }
    error.clear();
    return true;
}
