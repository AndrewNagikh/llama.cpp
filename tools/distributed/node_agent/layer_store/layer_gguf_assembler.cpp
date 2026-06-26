#include "layer_gguf_assembler.h"

#include "architecture/architecture_descriptor.h"
#include "descriptor_materialize.h"
#include "architecture/worker_requirement.h"
#include "dist_common.h"

#include "httplib.h"

#include <algorithm>
#include <cstdio>
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
    if (length == 0) {
        return false;
    }
    if (source_url.rfind("file://", 0) == 0) {
        const std::string path = source_url.substr(7);
        std::ifstream in(path, std::ios::binary);
        if (!in) {
            return false;
        }
        in.seekg(static_cast<std::streamoff>(offset), std::ios::beg);
        out.resize(static_cast<size_t>(length));
        in.read(reinterpret_cast<char *>(out.data()), static_cast<std::streamsize>(length));
        return static_cast<bool>(in);
    }

    httplib::Headers headers = {
        { "User-Agent", "distributed-llama-node-agent/0.1" },
        { "Accept", "*/*" },
    };
    const std::string token = dist_hf_token();
    if (!token.empty()) {
        headers.emplace("Authorization", "Bearer " + token);
    }

    const uint64_t end = offset + length - 1;
    char range_buf[80];
    snprintf(range_buf, sizeof(range_buf), "bytes=%llu-%llu",
            static_cast<unsigned long long>(offset),
            static_cast<unsigned long long>(end));
    headers.emplace("Range", range_buf);

    httplib::Client cli(source_url.c_str());
    cli.set_connection_timeout(30, 0);
    cli.set_read_timeout(300, 0);
    cli.set_follow_location(true);

    const auto res = cli.Get(source_url.c_str(), headers);
    if (!res || (res->status != 200 && res->status != 206)) {
        return false;
    }
    out.assign(res->body.begin(), res->body.end());
    return !out.empty();
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

bool layer_store_materialize_gguf(
        const layer_store & store,
        const model_manifest & manifest,
        const std::string & output_path,
        const int32_t layer_start,
        const int32_t layer_end,
        const bool include_embedding,
        const bool include_output) {
    if (manifest.empty()) {
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
        return false;
    }

    std::vector<uint8_t> file(file_size, 0);

    const auto metadata = store.load_metadata_blob();
    if (metadata.has_value() && !metadata->empty()) {
        const size_t n = std::min(metadata->size(), file.size());
        std::memcpy(file.data(), metadata->data(), n);
    }

    auto write_layer_blob = [&](const int32_t layer_index) -> bool {
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
    };

    for (int32_t layer = layer_start; layer < layer_end; ++layer) {
        if (!write_layer_blob(layer)) {
            return false;
        }
    }

    const auto desc = build_architecture_descriptor(manifest);
    const worker_role role = role_from_flags(include_embedding, include_output);
    const auto plan = materialize_plan_for_worker(
            desc.worker_requirements, role, layer_start, layer_end);

    std::vector<std::string> required_blobs = plan.required_blobs;
    if (include_embedding) {
        for (const semantic_blob & blob : desc.blobs) {
            if (blob.deploy == blob_deploy_target::entry_node ||
                    blob.deploy == blob_deploy_target::all_nodes) {
                required_blobs.push_back(blob.id);
            }
        }
    }
    if (include_output) {
        for (const semantic_blob & blob : desc.blobs) {
            if (blob.deploy == blob_deploy_target::final_node ||
                    blob.role == tensor_semantic_role::output_head ||
                    blob.role == tensor_semantic_role::output_norm) {
                required_blobs.push_back(blob.id);
            }
        }
    }

    std::sort(required_blobs.begin(), required_blobs.end());
    required_blobs.erase(
            std::unique(required_blobs.begin(), required_blobs.end()),
            required_blobs.end());

    std::string err;
    if (!materialize_descriptor_tensors(store, desc, required_blobs, file, err)) {
        return false;
    }

    std::error_code ec;
    std::filesystem::create_directories(std::filesystem::path(output_path).parent_path(), ec);

    std::ofstream out(output_path, std::ios::binary | std::ios::trunc);
    if (!out) {
        return false;
    }
    out.write(reinterpret_cast<const char *>(file.data()), static_cast<std::streamoff>(file.size()));
    return static_cast<bool>(out);
}
