#include "manifest_builder.h"

#include "../model_registry.h"

#include "dist_common.h"
#include "ggml.h"
#include "gguf.h"
#include "httplib.h"

#include <algorithm>
#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <stdexcept>

using json = nlohmann::json;

// ---------------------------------------------------------------------------
// Role helpers
// ---------------------------------------------------------------------------

std::string tensor_role_to_string(tensor_role role) {
    switch (role) {
        case tensor_role::embedding:    return "embedding";
        case tensor_role::output_norm: return "output_norm";
        case tensor_role::lm_head:      return "lm_head";
        case tensor_role::norm:         return "norm";
        case tensor_role::layer:        return "layer";
        case tensor_role::other:        return "other";
        case tensor_role::unknown:      return "unknown";
    }
    return "unknown";
}

tensor_role tensor_role_from_string(const std::string & s) {
    if (s == "embedding")     return tensor_role::embedding;
    if (s == "output_norm")   return tensor_role::output_norm;
    if (s == "lm_head")       return tensor_role::lm_head;
    if (s == "norm")          return tensor_role::norm;
    if (s == "layer")         return tensor_role::layer;
    if (s == "other")         return tensor_role::other;
    return tensor_role::unknown;
}

// ---------------------------------------------------------------------------
// JSON serialization
// ---------------------------------------------------------------------------

json tensor_descriptor::to_json() const {
    json dims_json = json::array();
    for (const uint64_t d : dims) {
        dims_json.push_back(d);
    }
    return {
        { "name",       name },
        { "ggml_type",  ggml_type },
        { "dims",       dims_json },
        { "size_bytes", size_bytes },
        { "offset",     offset },
        { "layer",      layer },
        { "role",       tensor_role_to_string(role) },
    };
}

tensor_descriptor tensor_descriptor::from_json(const json & j) {
    tensor_descriptor t;
    t.name       = j.value("name", "");
    t.ggml_type  = j.value("ggml_type", "");
    t.size_bytes = j.value("size_bytes", static_cast<uint64_t>(0));
    t.offset     = j.value("offset", static_cast<uint64_t>(0));
    t.layer      = j.value("layer", -1);
    t.role       = tensor_role_from_string(j.value("role", "unknown"));
    if (j.contains("dims") && j["dims"].is_array()) {
        for (const auto & d : j["dims"]) {
            t.dims.push_back(d.get<uint64_t>());
        }
    }
    return t;
}

json layer_descriptor::to_json() const {
    return {
        { "layer_index", layer_index },
        { "size_bytes",  size_bytes },
        { "tensors",     tensors },
    };
}

layer_descriptor layer_descriptor::from_json(const json & j) {
    layer_descriptor l;
    l.layer_index = j.value("layer_index", -1);
    l.size_bytes  = j.value("size_bytes", static_cast<uint64_t>(0));
    if (j.contains("tensors") && j["tensors"].is_array()) {
        for (const auto & t : j["tensors"]) {
            l.tensors.push_back(t.get<std::string>());
        }
    }
    return l;
}

bool model_manifest::empty() const {
    return architecture.empty() && tensors.empty();
}

json model_manifest::to_json() const {
    json tensors_json = json::array();
    for (const auto & t : tensors) {
        tensors_json.push_back(t.to_json());
    }

    json layers_json = json::array();
    for (const auto & l : layers) {
        layers_json.push_back(l.to_json());
    }

    return {
        { "architecture",        architecture },
        { "n_layer",             n_layer },
        { "n_ctx",               n_ctx },
        { "n_vocab",             n_vocab },
        { "n_embd",              n_embd },
        { "gguf_version",        gguf_version },
        { "tensors",             tensors_json },
        { "layers",              layers_json },
        { "special_tensors",     special_tensors },
        { "tensor_data_offset",  tensor_data_offset },
        { "metadata_bytes_read", metadata_bytes_read },
        { "source_file",         source_file },
    };
}

model_manifest model_manifest::from_json(const json & j) {
    model_manifest m;
    m.architecture        = j.value("architecture", "");
    m.n_layer             = j.value("n_layer", static_cast<uint32_t>(0));
    m.n_ctx               = j.value("n_ctx", static_cast<uint32_t>(0));
    m.n_vocab             = j.value("n_vocab", static_cast<uint32_t>(0));
    m.n_embd              = j.value("n_embd", static_cast<uint32_t>(0));
    m.gguf_version        = j.value("gguf_version", static_cast<uint32_t>(0));
    m.tensor_data_offset  = j.value("tensor_data_offset", static_cast<uint64_t>(0));
    m.metadata_bytes_read = j.value("metadata_bytes_read", static_cast<uint64_t>(0));
    m.source_file         = j.value("source_file", "");

    if (j.contains("tensors") && j["tensors"].is_array()) {
        for (const auto & item : j["tensors"]) {
            m.tensors.push_back(tensor_descriptor::from_json(item));
        }
    }
    if (j.contains("layers") && j["layers"].is_array()) {
        for (const auto & item : j["layers"]) {
            m.layers.push_back(layer_descriptor::from_json(item));
        }
    }
    if (j.contains("special_tensors") && j["special_tensors"].is_object()) {
        for (const auto & kv : j["special_tensors"].items()) {
            m.special_tensors[kv.key()] = kv.value().get<std::string>();
        }
    }
    return m;
}

// ---------------------------------------------------------------------------
// Tensor classification and layer aggregation
// ---------------------------------------------------------------------------

void classify_tensor(const std::string & name, int32_t & layer, tensor_role & role) {
    layer = -1;
    role  = tensor_role::other;

    if (name == "token_embd.weight") {
        role = tensor_role::embedding;
        return;
    }
    if (name == "output.weight") {
        role = tensor_role::lm_head;
        return;
    }
    if (name == "output_norm.weight") {
        role = tensor_role::output_norm;
        return;
    }
    if (name == "norm.weight" || name == "token_embd_norm.weight") {
        role = tensor_role::norm;
        return;
    }

    if (name.rfind("blk.", 0) == 0) {
        const size_t dot = name.find('.', 4);
        if (dot != std::string::npos) {
            try {
                layer = std::stoi(name.substr(4, dot - 4));
                role  = tensor_role::layer;
            } catch (...) {
                role = tensor_role::other;
            }
        }
    }
}

std::vector<layer_descriptor> build_layer_descriptors(const std::vector<tensor_descriptor> & tensors) {
    std::map<int32_t, layer_descriptor> by_layer;

    for (const auto & t : tensors) {
        if (t.layer < 0) {
            continue;
        }
        layer_descriptor & ld = by_layer[t.layer];
        if (ld.layer_index < 0) {
            ld.layer_index = t.layer;
        }
        ld.size_bytes += t.size_bytes;
        ld.tensors.push_back(t.name);
    }

    std::vector<layer_descriptor> layers;
    layers.reserve(by_layer.size());
    for (auto & kv : by_layer) {
        layers.push_back(std::move(kv.second));
    }
    std::sort(layers.begin(), layers.end(),
            [](const layer_descriptor & a, const layer_descriptor & b) {
                return a.layer_index < b.layer_index;
            });
    return layers;
}

// ---------------------------------------------------------------------------
// GGUF context -> manifest
// ---------------------------------------------------------------------------

namespace {

static uint32_t read_arch_u32(const gguf_context * ctx, const std::string & arch, const char * suffix) {
    const std::string key = arch + suffix;
    const int64_t id = gguf_find_key(ctx, key.c_str());
    if (id < 0) {
        return 0;
    }
    return gguf_get_val_u32(ctx, id);
}

static model_manifest manifest_from_gguf(gguf_context * ctx, ggml_context * ggml_ctx,
        const std::string & source_file, uint64_t bytes_read) {
    model_manifest manifest;
    manifest.source_file        = source_file;
    manifest.metadata_bytes_read = bytes_read;
    manifest.gguf_version       = gguf_get_version(ctx);
    manifest.tensor_data_offset = gguf_get_data_offset(ctx);

    const int64_t arch_id = gguf_find_key(ctx, "general.architecture");
    if (arch_id >= 0) {
        manifest.architecture = gguf_get_val_str(ctx, arch_id);
    }

    if (!manifest.architecture.empty()) {
        manifest.n_layer = read_arch_u32(ctx, manifest.architecture, ".block_count");
        manifest.n_ctx   = read_arch_u32(ctx, manifest.architecture, ".context_length");
        manifest.n_vocab = read_arch_u32(ctx, manifest.architecture, ".vocab_size");
        manifest.n_embd  = read_arch_u32(ctx, manifest.architecture, ".embedding_length");
    }

    const int64_t n_tensors = gguf_get_n_tensors(ctx);
    manifest.tensors.reserve((size_t) n_tensors);

    const uint64_t data_base = gguf_get_data_offset(ctx);

    for (int64_t i = 0; i < n_tensors; ++i) {
        tensor_descriptor td;
        td.name       = gguf_get_tensor_name(ctx, i);
        td.ggml_type  = ggml_type_name(gguf_get_tensor_type(ctx, i));
        td.size_bytes = gguf_get_tensor_size(ctx, i);
        td.offset     = data_base + gguf_get_tensor_offset(ctx, i);

        if (ggml_ctx) {
            const ggml_tensor * t = ggml_get_tensor(ggml_ctx, td.name.c_str());
            if (t) {
                const int n_dims = ggml_n_dims(t);
                td.dims.reserve((size_t) n_dims);
                for (int d = 0; d < n_dims; ++d) {
                    td.dims.push_back((uint64_t) t->ne[d]);
                }
            }
        }

        classify_tensor(td.name, td.layer, td.role);
        manifest.tensors.push_back(std::move(td));
    }

    for (const auto & t : manifest.tensors) {
        if (t.role == tensor_role::embedding) {
            manifest.special_tensors["embedding"] = t.name;
        } else if (t.role == tensor_role::lm_head) {
            manifest.special_tensors["lm_head"] = t.name;
        } else if (t.role == tensor_role::output_norm) {
            manifest.special_tensors["output_norm"] = t.name;
        } else if (t.role == tensor_role::norm) {
            manifest.special_tensors["norm"] = t.name;
        }
        if (manifest.n_vocab == 0 && t.name == "token_embd.weight" && t.dims.size() >= 2) {
            manifest.n_vocab = (uint32_t) t.dims[1];
        }
    }

    manifest.layers = build_layer_descriptors(manifest.tensors);

    if (manifest.n_layer == 0 && !manifest.layers.empty()) {
        manifest.n_layer = (uint32_t) manifest.layers.size();
    }

    return manifest;
}

static std::string resolve_local_gguf(
        const dist_model_record & record,
        const std::string & models_dir,
        const std::string & fallback_model_path) {
    if (!fallback_model_path.empty() &&
        std::filesystem::path(fallback_model_path).filename() == record.filename &&
        std::filesystem::exists(fallback_model_path)) {
        return fallback_model_path;
    }
    if (!models_dir.empty() && !record.filename.empty()) {
        const std::string candidate = (std::filesystem::path(models_dir) / record.filename).string();
        if (std::filesystem::exists(candidate)) {
            return candidate;
        }
    }
    return {};
}

static std::string find_download_url(const dist_model_record & record) {
    for (const auto & f : record.files) {
        if (f.filename == record.filename && !f.download_url.empty()) {
            return f.download_url;
        }
    }
    for (const auto & f : record.files) {
        if (f.filename.size() >= 5 &&
            f.filename.substr(f.filename.size() - 5) == ".gguf" &&
            !f.download_url.empty()) {
            return f.download_url;
        }
    }
    return {};
}

static bool http_range_fetch(const std::string & url, size_t end_inclusive, std::vector<uint8_t> & out) {
    httplib::Headers headers = {
        { "User-Agent", "distributed-llama-orchestrator/0.1" },
        { "Accept", "*/*" },
    };

    const std::string token = dist_hf_token();
    if (!token.empty()) {
        headers.emplace("Authorization", "Bearer " + token);
    }

    char range_buf[64];
    snprintf(range_buf, sizeof(range_buf), "bytes=0-%zu", end_inclusive);
    headers.emplace("Range", range_buf);

    httplib::Client cli(url.c_str());
    cli.set_connection_timeout(15, 0);
    cli.set_read_timeout(120, 0);
    cli.set_follow_location(true);

    const auto res = cli.Get(url.c_str(), headers);
    if (!res) {
        return false;
    }
    if (res->status != 200 && res->status != 206) {
        return false;
    }

    out.assign(res->body.begin(), res->body.end());
    return !out.empty();
}

static manifest_build_result build_manifest_from_url(const std::string & url, const std::string & source_file) {
    manifest_build_result result;
    result.manifest.source_file = source_file;

    size_t chunk = 256 * 1024;
    const size_t max_chunk = 16 * 1024 * 1024;

    while (chunk <= max_chunk) {
        std::vector<uint8_t> body;
        if (!http_range_fetch(url, chunk - 1, body)) {
            result.error = "HTTP range request failed for " + url;
            return result;
        }

        std::string parse_error;
        model_manifest manifest = build_manifest_from_buffer(body, parse_error);
        if (!parse_error.empty()) {
            if (chunk == max_chunk) {
                result.error = parse_error;
                return result;
            }
            chunk *= 2;
            continue;
        }

        const size_t meta_size = manifest.tensor_data_offset > 0
                ? (size_t) manifest.tensor_data_offset
                : body.size();

        if (meta_size > body.size()) {
            if (chunk == max_chunk) {
                result.error = "metadata section exceeds fetched range";
                return result;
            }
            chunk = std::max(chunk * 2, meta_size + 4096);
            continue;
        }

        result.success    = true;
        result.manifest   = std::move(manifest);
        result.manifest.source_file        = source_file;
        result.manifest.metadata_bytes_read = body.size();
        result.bytes_read = body.size();
        return result;
    }

    result.error = "metadata section exceeds 16MB fetch limit";
    return result;
}

} // namespace

model_manifest build_manifest_from_file(const std::string & path) {
    ggml_context * ggml_ctx = nullptr;
    gguf_init_params params{};
    params.no_alloc = true;
    params.ctx      = &ggml_ctx;

    gguf_context * ctx = gguf_init_from_file(path.c_str(), params);
    if (!ctx) {
        throw std::runtime_error("failed to parse GGUF file: " + path);
    }

    const uint64_t bytes_read = gguf_get_meta_size(ctx);
    model_manifest manifest   = manifest_from_gguf(ctx, ggml_ctx, path, bytes_read);
    if (ggml_ctx) {
        ggml_free(ggml_ctx);
    }
    gguf_free(ctx);
    return manifest;
}

model_manifest build_manifest_from_buffer(const std::vector<uint8_t> & data, std::string & error) {
    error.clear();
    if (data.size() < 4) {
        error = "buffer too small for GGUF header";
        return {};
    }

    if (memcmp(data.data(), "GGUF", 4) != 0) {
        error = "invalid GGUF magic";
        return {};
    }

    ggml_context * ggml_ctx = nullptr;
    gguf_init_params params{};
    params.no_alloc = true;
    params.ctx      = &ggml_ctx;

    gguf_context * ctx = gguf_init_from_buffer(data.data(), data.size(), params);
    if (!ctx) {
        error = "failed to parse GGUF metadata from buffer";
        return {};
    }

    model_manifest manifest = manifest_from_gguf(ctx, ggml_ctx, "", data.size());
    if (ggml_ctx) {
        ggml_free(ggml_ctx);
    }
    gguf_free(ctx);
    return manifest;
}

manifest_build_result build_manifest_for_record(
        const dist_model_record & record,
        const std::string & models_dir,
        const std::string & fallback_model_path) {
    manifest_build_result result;

    const std::string local = resolve_local_gguf(record, models_dir, fallback_model_path);
    if (!local.empty()) {
        try {
            result.manifest   = build_manifest_from_file(local);
            result.bytes_read = result.manifest.metadata_bytes_read;
            result.success    = true;
            return result;
        } catch (const std::exception & e) {
            result.error = e.what();
            return result;
        }
    }

    const std::string url = find_download_url(record);
    if (url.empty()) {
        result.error = "no local GGUF file and no download URL in discovery metadata";
        return result;
    }

    return build_manifest_from_url(url, record.filename);
}
