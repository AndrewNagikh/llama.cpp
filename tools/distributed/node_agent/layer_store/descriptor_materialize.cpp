#include "descriptor_materialize.h"

#include "architecture/semantic_blob.h"

#include <cstring>

namespace {

bool write_tensor_at(
        std::vector<uint8_t> & file,
        const uint64_t offset,
        const uint8_t * data,
        const size_t len) {
    if (offset + len > file.size()) {
        return false;
    }
    std::memcpy(file.data() + offset, data, len);
    return true;
}

const semantic_blob * resolve_blob(
        const architecture_descriptor & desc,
        const std::string & blob_id) {
    return find_blob(desc.blobs, blob_id);
}

std::string storage_blob_id(const semantic_blob & blob) {
    if (blob.storage_alias && !blob.storage_blob_id.empty()) {
        return blob.storage_blob_id;
    }
    return blob.id;
}

} // namespace

bool store_has_required_blobs(
        const layer_store & store,
        const architecture_descriptor & desc,
        const std::vector<std::string> & required_blobs) {
    for (const std::string & blob_id : required_blobs) {
        const semantic_blob * blob = resolve_blob(desc, blob_id);
        if (blob == nullptr || blob->tensors.empty()) {
            continue;
        }
        const std::string storage_id = storage_blob_id(*blob);
        if (!store.has_semantic_blob(storage_id, blob->tensors)) {
            return false;
        }
    }
    return true;
}

bool verify_required_blobs(
        const layer_store & store,
        const architecture_descriptor & desc,
        const std::vector<std::string> & required_blobs,
        std::string & err) {
    for (const std::string & blob_id : required_blobs) {
        const semantic_blob * blob = resolve_blob(desc, blob_id);
        if (blob == nullptr || blob->tensors.empty()) {
            continue;
        }
        const std::string storage_id = storage_blob_id(*blob);
        for (const auto & slot : blob->tensors) {
            const std::string checksum = "manifest:tensor:" + slot.name;
            if (!store.has_blob_tensor(storage_id, slot.name)) {
                err = "missing blob tensor " + storage_id + "/" + slot.name;
                return false;
            }
            if (!store.verify_blob_tensor(storage_id, slot.name, checksum)) {
                err = "checksum failed for " + storage_id + "/" + slot.name;
                return false;
            }
        }
    }
    err.clear();
    return true;
}

bool materialize_descriptor_tensors(
        const layer_store & store,
        const architecture_descriptor & desc,
        const std::vector<std::string> & required_blobs,
        std::vector<uint8_t> & file,
        std::string & err) {
    for (const std::string & blob_id : required_blobs) {
        const semantic_blob * blob = resolve_blob(desc, blob_id);
        if (blob == nullptr) {
            continue;
        }
        const std::string storage_id = storage_blob_id(*blob);
        for (const auto & slot : blob->tensors) {
            std::vector<uint8_t> data;
            if (!store.load_blob_tensor(storage_id, slot.name, data) || data.empty()) {
                err = "failed to load blob tensor " + storage_id + "/" + slot.name;
                return false;
            }
            const uint64_t offset = slot.offset > 0 ? slot.offset : 0;
            if (!write_tensor_at(file, offset, data.data(), data.size())) {
                err = "failed to write tensor at offset for " + slot.name;
                return false;
            }
        }
    }
    err.clear();
    return true;
}
