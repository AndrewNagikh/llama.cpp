#include "layer_store_populate.h"

#include "architecture/architecture_descriptor.h"
#include "architecture/semantic_blob.h"
#include "orchestrator/install_planner/install_planner.h"
#include "verification_common.h"

namespace {

bool store_blob_tensors(
        layer_store & store,
        const semantic_blob & blob,
        const std::string & source_path) {
    if (blob.storage_alias) {
        return true;
    }

    const std::string storage_id = blob.storage_blob_id.empty() ? blob.id : blob.storage_blob_id;
    for (const auto & slot : blob.tensors) {
        if (slot.size_bytes == 0) {
            continue;
        }
        std::vector<uint8_t> data;
        if (!read_file_range(source_path, slot.offset, slot.size_bytes, data)) {
            return false;
        }
        const std::string checksum = "manifest:tensor:" + slot.name;
        if (!store.store_blob_tensor(
                    storage_id,
                    slot.name,
                    data.data(),
                    data.size(),
                    slot.offset,
                    checksum)) {
            return false;
        }
    }
    return true;
}

} // namespace

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

    const auto desc = build_architecture_descriptor(manifest);
    for (const auto & blob : desc.blobs) {
        if (!store_blob_tensors(store, blob, source_path)) {
            return false;
        }
    }

    for (const auto & ld : manifest.layers) {
        const layer_byte_range lr = manifest_layer_byte_range(manifest, ld.layer_index);
        if (lr.length == 0) {
            continue;
        }
        std::vector<uint8_t> data;
        if (!read_file_range(source_path, lr.offset, lr.length, data)) {
            return false;
        }
        if (!store.store_layer(
                    ld.layer_index,
                    data.data(),
                    data.size(),
                    lr.offset,
                    lr.offset + lr.length,
                    lr.checksum)) {
            return false;
        }
    }

    return true;
}
