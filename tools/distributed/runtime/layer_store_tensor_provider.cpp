#include "layer_store_tensor_provider.h"

#include "architecture/semantic_blob.h"

#include <optional>

layer_store_tensor_provider::layer_store_tensor_provider(
        layer_store store,
        semantic_runtime_descriptor rt)
        : store_(std::move(store)), rt_(std::move(rt)) {}

bool layer_store_tensor_provider::resolve_blob_tensor(
        const std::string & tensor_name,
        std::string & blob_id,
        std::string & blob_tensor) const {
    for (const auto & blob : rt_.blobs) {
        for (const auto & slot : blob.tensors) {
            if (slot.name == tensor_name) {
                blob_id     = blob.id;
                blob_tensor = slot.name;
                return true;
            }
        }
    }
    return false;
}

bool layer_store_tensor_provider::tensor_exists(const std::string & tensor_name) const {
    std::string blob_id;
    std::string blob_tensor;
    if (!resolve_blob_tensor(tensor_name, blob_id, blob_tensor)) {
        return false;
    }
    return store_.has_blob_tensor(blob_id, blob_tensor);
}

bool layer_store_tensor_provider::load_tensor(
        const std::string & tensor_name,
        std::vector<uint8_t> & out) const {
    std::string blob_id;
    std::string blob_tensor;
    if (!resolve_blob_tensor(tensor_name, blob_id, blob_tensor)) {
        return false;
    }
    return store_.load_blob_tensor(blob_id, blob_tensor, out);
}

bool layer_store_tensor_provider::verify_tensor(
        const std::string & tensor_name,
        const std::string & expected_checksum) const {
    std::string blob_id;
    std::string blob_tensor;
    if (!resolve_blob_tensor(tensor_name, blob_id, blob_tensor)) {
        return false;
    }
    return store_.verify_blob_tensor(blob_id, blob_tensor, expected_checksum);
}

uint64_t layer_store_tensor_provider::tensor_size_bytes(const std::string & tensor_name) const {
    std::string blob_id;
    std::string blob_tensor;
    if (!resolve_blob_tensor(tensor_name, blob_id, blob_tensor)) {
        return 0;
    }
    for (const auto & info : store_.list_blob_tensors()) {
        if (info.blob_id == blob_id && info.tensor_name == blob_tensor) {
            return info.size_bytes;
        }
    }
    std::vector<uint8_t> data;
    if (!store_.load_blob_tensor(blob_id, blob_tensor, data)) {
        return 0;
    }
    return static_cast<uint64_t>(data.size());
}

std::unique_ptr<layer_store_tensor_provider> make_layer_store_tensor_provider(
        const std::string & models_dir,
        const std::string & model_id,
        const model_manifest & manifest) {
    layer_store store(models_dir, model_id);
    const semantic_runtime_descriptor rt = build_semantic_runtime_descriptor(manifest);
    return std::make_unique<layer_store_tensor_provider>(std::move(store), std::move(rt));
}
