#include "layer_gguf_assembler.h"

#include "architecture/semantic_runtime_descriptor.h"
#include "descriptor_materialize.h"
#include "dist_http_fetch.h"
#include "worker_builder.h"

#include <algorithm>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <vector>

namespace {

static bool fetch_bytes(
        const std::string & source_url,
        const uint64_t offset,
        const uint64_t length,
        std::vector<uint8_t> & out) {
    return dist_http_get_range(source_url, offset, length, out);
}

static bool write_at(std::vector<uint8_t> & file, const uint64_t offset, const uint8_t * data, const size_t len) {
    if (offset + len > file.size()) {
        return false;
    }
    std::memcpy(file.data() + offset, data, len);
    return true;
}

static worker_role role_from_flags(const bool include_embedding, const bool include_output) {
    if (include_embedding && include_output) {
        return worker_role::full;
    }
    if (include_embedding) {
        return worker_role::entry;
    }
    if (include_output) {
        return worker_role::final;
    }
    return worker_role::middle;
}

} // namespace

bool layer_store_cache_metadata(
        layer_store & store,
        const model_manifest & manifest,
        const std::string & source_url) {
    if (source_url.empty() || manifest.tensor_data_offset == 0) {
        return false;
    }

    const auto existing = store.metadata_bytes();
    if (existing.has_value() && *existing == manifest.tensor_data_offset) {
        return true;
    }

    std::vector<uint8_t> meta;
    if (!fetch_bytes(source_url, 0, manifest.tensor_data_offset, meta)) {
        return false;
    }
    return store.save_metadata_blob(meta);
}

bool layer_store_materialize_tokenizer_shell(
        const layer_store & store,
        const model_manifest & manifest,
        const std::string & output_path) {
    if (manifest.empty() || manifest.tensor_data_offset == 0) {
        return false;
    }
    const auto metadata = store.load_metadata_blob();
    if (!metadata.has_value() || metadata->empty()) {
        return false;
    }

    std::vector<uint8_t> file(manifest.tensor_data_offset, 0);
    const size_t n = std::min(metadata->size(), file.size());
    std::memcpy(file.data(), metadata->data(), n);

    std::ofstream out(output_path, std::ios::binary | std::ios::trunc);
    if (!out) {
        return false;
    }
    out.write(reinterpret_cast<const char *>(file.data()), static_cast<std::streamoff>(file.size()));
    return static_cast<bool>(out);
}

bool layer_store_materialize_gguf(
        const layer_store & store,
        const model_manifest & manifest,
        const std::string & output_path,
        const int32_t layer_start,
        const int32_t layer_end,
        const bool include_embedding,
        const bool include_output) {
    const worker_role role = role_from_flags(include_embedding, include_output);
    const semantic_runtime_descriptor rt = build_semantic_runtime_descriptor(manifest);
    std::string err;
    return materialize_worker_gguf(
            store, manifest, rt, role, layer_start, layer_end, output_path, err);
}
