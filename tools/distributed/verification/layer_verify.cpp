#include "layer_verify.h"

#include "node_agent/layer_store/layer_checksum.h"
#include "node_agent/layer_store/layer_special.h"
#include "orchestrator/install_planner/install_planner.h"
#include "verification_common.h"

#include <set>

layer_verify_result layer_verify_store(
        const layer_store & store,
        const model_manifest & manifest,
        const std::string & source_path) {
    layer_verify_result result;
    result.summary.name = "layer_store";

    if (manifest.empty()) {
        result.summary.status  = verify_status::fail;
        result.summary.message = "empty manifest";
        return result;
    }

    const auto metadata = store.metadata_bytes();
    if (!metadata.has_value() || *metadata != manifest.tensor_data_offset) {
        result.issues.push_back("metadata.bin missing or wrong size (expected " +
                std::to_string(manifest.tensor_data_offset) + ")");
    }

    std::set<int32_t> expected_layers;
    for (const auto & ld : manifest.layers) {
        expected_layers.insert(ld.layer_index);
    }

    for (const int32_t layer : expected_layers) {
        if (!store.has_layer(layer)) {
            result.issues.push_back("missing layer blob: " + std::to_string(layer));
            continue;
        }
        const auto blob = store.get_layer(layer);
        if (!blob.has_value()) {
            result.issues.push_back("cannot read layer metadata: " + std::to_string(layer));
            continue;
        }
        const layer_byte_range expected = manifest_layer_byte_range(manifest, layer);
        if (expected.length > 0) {
            if (blob->offset_begin != expected.offset || blob->size_bytes != expected.length) {
                result.issues.push_back("layer " + std::to_string(layer) + " offset/size mismatch");
            }
        }
        if (!store.verify_layer(layer, "manifest:layer:" + std::to_string(layer))) {
            std::vector<uint8_t> data;
            if (store.load_layer(layer, data) && !source_path.empty()) {
                std::vector<uint8_t> orig;
                if (read_file_range(source_path, blob->offset_begin, blob->size_bytes, orig) && orig != data) {
                    result.issues.push_back("layer " + std::to_string(layer) + " bytes differ from source");
                } else {
                    result.issues.push_back("layer " + std::to_string(layer) + " checksum verify failed");
                }
            } else {
                result.issues.push_back("layer " + std::to_string(layer) + " checksum verify failed");
            }
        }
    }

    if (!store.has_layer(layer_special::embedding)) {
        result.issues.push_back("missing embedding blob (-1)");
    }
    if (!store.has_layer(layer_special::output)) {
        result.issues.push_back("missing output blob (-2)");
    }

    result.summary.status  = result.issues.empty() ? verify_status::ok : verify_status::fail;
    result.summary.message = result.issues.empty() ? "layer store matches manifest" : "layer store issues found";
    result.summary.details = { { "issue_count", result.issues.size() } };
    return result;
}
