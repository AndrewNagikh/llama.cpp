#include "semantic_runtime_descriptor.h"

#include "architecture_plugin.h"
#include "tensor_graph.h"

namespace {

worker_descriptor to_worker_descriptor(const worker_requirement & req) {
    worker_descriptor wd;
    wd.role              = req.role;
    wd.required_blob_ids = req.required_blobs;
    return wd;
}

architecture_descriptor architecture_from_runtime(const semantic_runtime_descriptor & rt) {
    architecture_descriptor desc;
    desc.architecture     = rt.architecture;
    desc.family           = rt.family;
    desc.tied_embeddings  = rt.tied_embeddings;
    desc.separate_lm_head = rt.separate_lm_head;
    desc.is_moe           = rt.is_moe;
    desc.blobs            = rt.blobs;
    for (const auto & w : rt.workers) {
        worker_requirement req;
        req.role           = w.role;
        req.required_blobs = w.required_blob_ids;
        desc.worker_requirements.push_back(std::move(req));
    }
    return desc;
}

} // namespace

semantic_runtime_descriptor build_semantic_runtime_descriptor(const model_manifest & manifest) {
    const tensor_graph graph = analyze_tensor_graph(manifest);
    const architecture_descriptor arch = build_descriptor_from_graph(graph);
    const architecture_plugin & plugin = select_architecture_plugin(manifest);

    semantic_runtime_descriptor rt;
    rt.architecture     = arch.architecture;
    rt.family           = plugin.family();
    rt.tied_embeddings  = arch.tied_embeddings;
    rt.separate_lm_head = arch.separate_lm_head;
    rt.is_moe           = arch.is_moe;
    rt.blobs            = arch.blobs;

    for (const auto & req : arch.worker_requirements) {
        rt.workers.push_back(to_worker_descriptor(req));
    }
    return rt;
}

const worker_descriptor * find_worker_descriptor(
        const semantic_runtime_descriptor & rt,
        const worker_role role) {
    for (const auto & w : rt.workers) {
        if (w.role == role) {
            return &w;
        }
    }
    return nullptr;
}

worker_materialize_plan worker_materialize_plan_for_role(
        const semantic_runtime_descriptor & rt,
        const worker_role role,
        const int32_t layer_start,
        const int32_t layer_end) {
    const architecture_descriptor arch = architecture_from_runtime(rt);
    return materialize_plan_for_worker(
            arch.worker_requirements, role, layer_start, layer_end);
}
