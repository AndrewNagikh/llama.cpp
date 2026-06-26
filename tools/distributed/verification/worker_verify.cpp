#include "worker_verify.h"

#include "architecture/architecture_descriptor.h"
#include "node_agent/layer_store/descriptor_materialize.h"
#include "architecture/worker_requirement.h"

bool verify_worker_materialization(
        const layer_store & store,
        const model_manifest & manifest,
        const int32_t layer_start,
        const int32_t layer_end,
        const bool include_embedding,
        const bool include_output,
        std::string & err) {
    const auto metadata = store.metadata_bytes();
    if (!metadata.has_value() || *metadata != manifest.tensor_data_offset) {
        err = "metadata checksum/size mismatch";
        return false;
    }

    const auto desc = build_architecture_descriptor(manifest);
    const worker_role role = include_embedding && include_output
            ? worker_role::full
            : include_embedding
                    ? worker_role::entry
                    : include_output ? worker_role::final : worker_role::middle;

    const auto plan = materialize_plan_for_worker(
            desc.worker_requirements, role, layer_start, layer_end);

    std::vector<std::string> required_blobs = plan.required_blobs;
    if (include_embedding) {
        required_blobs.push_back("embedding");
    }
    if (include_output) {
        if (find_blob(desc.blobs, "output_norm")) {
            required_blobs.push_back("output_norm");
        }
        if (find_blob(desc.blobs, "output_head")) {
            required_blobs.push_back("output_head");
        }
    }

    if (!verify_required_blobs(store, desc, required_blobs, err)) {
        return false;
    }

    for (int32_t layer = layer_start; layer < layer_end; ++layer) {
        if (!store.has_layer(layer)) {
            err = "missing layer " + std::to_string(layer);
            return false;
        }
        if (!store.verify_layer(layer, "manifest:layer:" + std::to_string(layer))) {
            err = "layer checksum failed: " + std::to_string(layer);
            return false;
        }
    }

    err.clear();
    return true;
}
