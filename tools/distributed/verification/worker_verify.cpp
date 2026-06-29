#include "worker_verify.h"

#include "architecture/semantic_runtime_descriptor.h"
#include "architecture/semantic_blob.h"
#include "node_agent/layer_store/descriptor_materialize.h"

namespace {

architecture_descriptor architecture_from_runtime(const semantic_runtime_descriptor & rt) {
    architecture_descriptor desc;
    desc.architecture     = rt.architecture;
    desc.family           = rt.family;
    desc.tied_embeddings  = rt.tied_embeddings;
    desc.separate_lm_head = rt.separate_lm_head;
    desc.is_moe           = rt.is_moe;
    desc.blobs            = rt.blobs;
    for (const worker_descriptor & w : rt.workers) {
        worker_requirement req;
        req.role           = w.role;
        req.required_blobs = w.required_blob_ids;
        desc.worker_requirements.push_back(std::move(req));
    }
    return desc;
}

} // namespace

bool verify_worker_materialization_for_role(
        const layer_store & store,
        const model_manifest & manifest,
        const worker_role role,
        const int32_t layer_start,
        const int32_t layer_end,
        std::string & err) {
    const auto metadata = store.metadata_bytes();
    if (!metadata.has_value() || *metadata != manifest.tensor_data_offset) {
        err = "metadata checksum/size mismatch";
        return false;
    }

    const semantic_runtime_descriptor rt = build_semantic_runtime_descriptor(manifest);
    const worker_materialize_plan plan =
            worker_materialize_plan_for_role(rt, role, layer_start, layer_end);

    std::vector<std::string> required_blobs = plan.required_blobs;
    for (int32_t layer = layer_start; layer < layer_end; ++layer) {
        required_blobs.push_back(layer_blob_id(layer));
    }

    const architecture_descriptor desc = architecture_from_runtime(rt);
    if (!verify_required_blobs(store, desc, required_blobs, err)) {
        return false;
    }

    err.clear();
    return true;
}

bool verify_worker_materialization(
        const layer_store & store,
        const model_manifest & manifest,
        const int32_t layer_start,
        const int32_t layer_end,
        const bool include_embedding,
        const bool include_output,
        std::string & err) {
    const worker_role role = include_embedding && include_output
            ? worker_role::full
            : include_embedding
                    ? worker_role::entry
                    : include_output ? worker_role::final : worker_role::middle;
    return verify_worker_materialization_for_role(
            store, manifest, role, layer_start, layer_end, err);
}
