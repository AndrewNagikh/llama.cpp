#include "worker_verify.h"

#include "architecture_descriptor/architecture_descriptor.h"
#include "node_agent/layer_store/layer_special.h"

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

    if (include_embedding) {
        if (!store.has_layer(layer_special::embedding)) {
            err = "missing embedding layer blob";
            return false;
        }
        if (!store.verify_layer(layer_special::embedding, "manifest:role:preamble") &&
                !store.verify_layer(layer_special::embedding, "manifest:role:embedding")) {
            err = "embedding layer checksum failed";
            return false;
        }
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

    if (include_output) {
        const auto desc = build_architecture_descriptor(manifest);
        if (architecture_materialize_needs_embedding_for_output(
                    desc, include_embedding, include_output)) {
            if (!store.has_layer(layer_special::embedding)) {
                err = "missing tied output embedding blob";
                return false;
            }
            if (!store.verify_layer(layer_special::embedding, "manifest:role:preamble") &&
                    !store.verify_layer(layer_special::embedding, "manifest:role:embedding")) {
                err = "tied output embedding checksum failed";
                return false;
            }
        }
        if (!store.has_layer(layer_special::output)) {
            err = "missing output layer blob";
            return false;
        }
        if (!store.verify_layer(layer_special::output, "manifest:role:output")) {
            err = "output layer checksum failed";
            return false;
        }
    }

    err.clear();
    return true;
}
