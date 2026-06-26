#include "layer_store_populate.h"

#include "node_agent/layer_store/layer_special.h"
#include "orchestrator/install_planner/install_planner.h"
#include "verification_common.h"

bool populate_layer_store_from_gguf(
        layer_store & store,
        const model_manifest & manifest,
        const std::string & source_path) {
    if (manifest.empty() || source_path.empty()) {
        return false;
    }

    std::vector<uint8_t> meta;
    if (!read_file_range(source_path, 0, manifest.tensor_data_offset, meta)) {
        return false;
    }
    if (!store.save_metadata_blob(meta)) {
        return false;
    }
    if (!store.save_manifest(manifest)) {
        return false;
    }

    auto store_range = [&](const int32_t layer_index, const layer_byte_range & range) -> bool {
        if (range.length == 0) {
            return true;
        }
        std::vector<uint8_t> data;
        if (!read_file_range(source_path, range.offset, range.length, data)) {
            return false;
        }
        return store.store_layer(
                layer_index,
                data.data(),
                data.size(),
                range.offset,
                range.offset + range.length,
                range.checksum);
    };

    const layer_byte_range preamble = manifest_global_preamble_range(manifest);
    if (preamble.length > 0) {
        if (!store_range(layer_special::embedding, preamble)) {
            return false;
        }
    } else {
        const layer_byte_range emb = manifest_role_byte_range(manifest, tensor_role::embedding);
        if (!store_range(layer_special::embedding, emb)) {
            return false;
        }
    }

    for (const auto & ld : manifest.layers) {
        const layer_byte_range lr = manifest_layer_byte_range(manifest, ld.layer_index);
        if (!store_range(ld.layer_index, lr)) {
            return false;
        }
    }

    layer_byte_range out_range{};
    const layer_byte_range norm = manifest_role_byte_range(manifest, tensor_role::output_norm);
    const layer_byte_range head = manifest_role_byte_range(manifest, tensor_role::lm_head);
    if (norm.length > 0) {
        out_range = norm;
    }
    if (head.length > 0) {
        if (out_range.length == 0) {
            out_range = head;
        } else {
            const uint64_t end = std::max(head.offset + head.length, out_range.offset + out_range.length);
            out_range.offset = std::min(head.offset, out_range.offset);
            out_range.length = end - out_range.offset;
            out_range.size_bytes = out_range.length;
        }
    }
    out_range.checksum = "manifest:role:output";
    if (!store_range(layer_special::output, out_range)) {
        return false;
    }

    return true;
}
