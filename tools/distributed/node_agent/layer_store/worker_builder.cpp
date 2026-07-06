#include "worker_builder.h"

#include "architecture/semantic_runtime_descriptor.h"
#include "descriptor_materialize.h"
#include "layer_gguf_assembler.h"

#include <algorithm>
#include <cstring>
#include <fstream>
#include <vector>

namespace {

bool write_at(std::vector<uint8_t> & file, const uint64_t offset, const uint8_t * data, const size_t len) {
    if (offset + len > file.size()) {
        return false;
    }
    std::memcpy(file.data() + offset, data, len);
    return true;
}

bool write_layer_merged(
        const layer_store & store,
        const int32_t layer_index,
        std::vector<uint8_t> & file) {
    if (!store.has_layer(layer_index)) {
        return false;
    }
    const auto blob = store.get_layer(layer_index);
    if (!blob.has_value()) {
        return false;
    }
    std::vector<uint8_t> data;
    if (!store.load_layer(layer_index, data) || data.empty()) {
        return false;
    }
    const uint64_t offset = blob->offset_begin > 0 ? blob->offset_begin : 0;
    return write_at(file, offset, data.data(), data.size());
}

bool write_layer_blob_tensors(
        const layer_store & store,
        const semantic_runtime_descriptor & rt,
        const int32_t layer_index,
        std::vector<uint8_t> & file,
        std::string & err) {
    const std::string blob_id = layer_blob_id(layer_index);
    const semantic_blob * blob = find_blob(rt.blobs, blob_id);
    if (blob == nullptr || blob->tensors.empty()) {
        err = "missing layer blob descriptor for layer " + std::to_string(layer_index);
        return false;
    }
    architecture_descriptor arch;
    arch.blobs = rt.blobs;
        return materialize_descriptor_tensors(store, arch, { blob_id }, file, err);
}

bool all_nodes_blob_required_for_role(const semantic_blob & blob, const worker_role role) {
    if (blob.deploy != blob_deploy_target::all_nodes) {
        return false;
    }
    // Tied embeddings mark embedding as all_nodes for mono parity; distributed
    // middle stages must not pull embedding weights onto every pipeline node.
    if (blob.id == "embedding" || blob.role == tensor_semantic_role::embedding) {
        return role == worker_role::embedding ||
               role == worker_role::full;
    }
    return true;
}

} // namespace

bool materialize_worker_gguf(
        const layer_store & store,
        const model_manifest & manifest,
        const semantic_runtime_descriptor & rt,
        const worker_role role,
        const int32_t layer_start,
        const int32_t layer_end,
        const std::string & output_path,
        std::string & err) {
    if (manifest.empty()) {
        err = "empty manifest";
        return false;
    }

    uint64_t file_size = manifest.tensor_data_offset;
    for (const auto & t : manifest.tensors) {
        if (t.offset > 0 || t.size_bytes > 0) {
            file_size = std::max(file_size, t.offset + t.size_bytes);
        }
    }
    for (const auto & blob : store.list_layers()) {
        if (blob.offset_end > file_size) {
            file_size = blob.offset_end;
        }
    }
    if (file_size == 0) {
        err = "cannot determine GGUF file size";
        return false;
    }

    std::vector<uint8_t> file(file_size, 0);

    const auto metadata = store.load_metadata_blob();
    if (metadata.has_value() && !metadata->empty()) {
        const size_t n = std::min(metadata->size(), file.size());
        std::memcpy(file.data(), metadata->data(), n);
    }

    for (int32_t layer = layer_start; layer < layer_end; ++layer) {
        const std::string layer_id = layer_blob_id(layer);
        const semantic_blob * layer_blob = find_blob(rt.blobs, layer_id);
        bool layer_ok = false;
        if (layer_blob != nullptr &&
                store.has_semantic_blob(layer_id, layer_blob->tensors)) {
            layer_ok = write_layer_blob_tensors(store, rt, layer, file, err);
        }
        if (!layer_ok) {
            if (!write_layer_merged(store, layer, file)) {
                if (err.empty()) {
                    err = "missing layer " + std::to_string(layer);
                }
                return false;
            }
        }
    }

    const worker_materialize_plan plan =
            worker_materialize_plan_for_role(rt, role, layer_start, layer_end);

    std::vector<std::string> required_blobs = plan.required_blobs;

    for (const semantic_blob & blob : rt.blobs) {
        if (all_nodes_blob_required_for_role(blob, role)) {
            required_blobs.push_back(blob.id);
        }
    }

    std::sort(required_blobs.begin(), required_blobs.end());
    required_blobs.erase(
            std::unique(required_blobs.begin(), required_blobs.end()),
            required_blobs.end());

    architecture_descriptor arch;
    arch.blobs = rt.blobs;
    if (!materialize_descriptor_tensors(store, arch, required_blobs, file, err)) {
        return false;
    }

    std::ofstream out(output_path, std::ios::binary | std::ios::trunc);
    if (!out) {
        err = "failed to open output path";
        return false;
    }
    out.write(reinterpret_cast<const char *>(file.data()), static_cast<std::streamoff>(file.size()));
    if (!out) {
        err = "failed to write GGUF";
        return false;
    }
    return true;
}
