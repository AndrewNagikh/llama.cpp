#include "layer_gguf_assembler.h"

#include "architecture/semantic_runtime_descriptor.h"
#include "descriptor_materialize.h"
#include "worker_builder.h"
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
