#include "memory_estimator.h"

#include "llama.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>

static constexpr uint64_t BYTES_PER_GB         = 1024ULL * 1024ULL * 1024ULL;
static constexpr uint64_t KV_ELEMENT_BYTES     = 2;   // fp16/bf16 KV cache cells.
static constexpr uint64_t MIN_COMPUTE_BYTES    = 256ULL * 1024ULL * 1024ULL;
static constexpr uint64_t MIN_SCRATCH_BYTES    = 128ULL * 1024ULL * 1024ULL;

static double bytes_to_gb(const uint64_t bytes) {
    return static_cast<double>(bytes) / static_cast<double>(BYTES_PER_GB);
}

double model_memory_requirements::weights_gb() const { return bytes_to_gb(weights_bytes); }
double model_memory_requirements::kv_gb()      const { return bytes_to_gb(kv_bytes); }
double model_memory_requirements::compute_gb() const { return bytes_to_gb(compute_bytes); }
double model_memory_requirements::scratch_gb() const { return bytes_to_gb(scratch_bytes); }
double model_memory_requirements::total_gb()   const { return bytes_to_gb(total_bytes()); }

static std::string dist_model_meta_str(const llama_model * model, const char * key) {
    char buf[256] = { 0 };
    const int32_t n = llama_model_meta_val_str(model, key, buf, sizeof(buf) - 1);
    if (n < 0) {
        return "";
    }
    return std::string(buf);
}

static int64_t dist_meta_int(const llama_model * model, const char * key, int64_t fallback) {
    const std::string val = dist_model_meta_str(model, key);
    if (val.empty()) {
        return fallback;
    }
    try {
        return static_cast<int64_t>(std::stoll(val));
    } catch (...) {
        return fallback;
    }
}

static uint64_t dist_file_size(const std::string & path) {
    std::error_code ec;
    const auto size = std::filesystem::file_size(path, ec);
    return ec ? 0ULL : static_cast<uint64_t>(size);
}

model_memory_requirements estimate_model_memory(
        const std::string & gguf_path,
        const int32_t       n_ctx) {
    model_memory_requirements result{};
    result.n_ctx = n_ctx;

    if (!std::filesystem::exists(gguf_path)) {
        return result;
    }

    ggml_backend_load_all();
    llama_model_params mparams = llama_model_default_params();
    llama_model * model = llama_model_load_from_file(gguf_path.c_str(), mparams);
    if (!model) {
        fprintf(stderr, "memory_estimator: failed to load %s for metadata\n", gguf_path.c_str());
        return result;
    }

    const int32_t n_layer   = llama_model_n_layer(model);
    const int32_t n_embd    = llama_model_n_embd(model);
    const int32_t n_head    = llama_model_n_head(model);
    int32_t       n_head_kv = llama_model_n_head_kv(model);
    if (n_head_kv <= 0) {
        n_head_kv = n_head > 0 ? n_head : 1;
    }

    // Try to read architecture-specific head dimension.
    const std::string arch = dist_model_meta_str(model, "general.architecture");
    int32_t head_dim = 0;
    if (!arch.empty()) {
        const std::string key = arch + ".attention.key_length";
        head_dim = static_cast<int32_t>(dist_meta_int(model, key.c_str(), 0));
    }
    if (head_dim <= 0 && n_head > 0) {
        head_dim = n_embd / n_head;
    }
    if (head_dim <= 0) {
        head_dim = n_embd > 0 ? n_embd : 1;
    }

    const uint64_t file_size = dist_file_size(gguf_path);
    result.model_id    = arch.empty() ? "unknown" : arch;
    result.n_layer     = n_layer;
    result.n_embd      = n_embd;
    result.n_ctx       = n_ctx;
    result.weights_bytes = file_size > 0 ? file_size : 0;

    // Per-layer KV increment: one K and one V vector per KV head per token.
    const uint64_t kv_per_token_per_layer = static_cast<uint64_t>(2) *
                                            static_cast<uint64_t>(n_head_kv) *
                                            static_cast<uint64_t>(head_dim) *
                                            KV_ELEMENT_BYTES;
    const uint64_t kv_per_layer_for_ctx = kv_per_token_per_layer * static_cast<uint64_t>(n_ctx);
    result.kv_bytes = kv_per_layer_for_ctx * static_cast<uint64_t>(std::max(1, n_layer));

    // Compute and scratch are conservative, architecture-agnostic placeholders.
    result.compute_bytes = std::max(MIN_COMPUTE_BYTES, result.weights_bytes / 16);
    result.scratch_bytes = std::max(MIN_SCRATCH_BYTES, result.weights_bytes / 32);

    // Build per-layer descriptors. Task 9 will replace weight_bytes with
    // real layer sizes; until then we spread the total evenly.
    const uint64_t per_layer_weight = n_layer > 0 ? (result.weights_bytes / static_cast<uint64_t>(n_layer)) : 0;
    result.layers.reserve(n_layer);
    for (int32_t i = 0; i < n_layer; ++i) {
        model_layer_memory layer{};
        layer.layer_index      = i;
        layer.weight_bytes     = per_layer_weight;
        layer.kv_bytes_per_token = kv_per_token_per_layer;
        result.layers.push_back(layer);
    }

    llama_model_free(model);
    return result;
}

nlohmann::json cluster_memory_fits_result::to_json() const {
    using json = nlohmann::json;
    json j = {
        { "fits", fits },
        { "required_gb", required_gb },
        { "available_gb", available_gb },
        { "missing_gb", missing_gb },
    };
    json warnings_arr = json::array();
    for (const auto & w : warnings) {
        warnings_arr.push_back(w);
    }
    j["warnings"] = warnings_arr;
    return j;
}

cluster_memory_fits_result dist_check_cluster_memory_fit(
        const model_memory_requirements & mem,
        const std::vector<dist_node_info> & nodes) {
    cluster_memory_fits_result result{};
    result.required_gb = mem.total_gb();

    double available_gb = 0.0;
    for (const auto & node : nodes) {
        if (!node.online) {
            continue;
        }
        // Primary execution budget: GPU first, CPU otherwise.
        const uint64_t budget = node.memory.has_gpu ? node.memory.free_vram_bytes
                                                    : node.memory.free_ram_bytes;
        available_gb += dist_bytes_to_gb(budget);
    }
    result.available_gb = available_gb;
    const double missing = result.required_gb - result.available_gb;
    result.missing_gb    = missing > 0.0 ? missing : 0.0;
    result.fits          = result.missing_gb <= 0.0;
    return result;
}

model_memory_requirements estimate_model_memory_from_catalog(
        const model_info & model,
        const int32_t      n_ctx) {
    model_memory_requirements result{};
    result.model_id = model.id;
    result.n_layer  = model.n_layers;
    result.n_embd   = model.n_embd;
    result.n_ctx    = n_ctx;

    const int32_t n_layer = std::max(1, model.n_layers);
    const int32_t n_embd  = std::max(1, model.n_embd);

    result.weights_bytes = static_cast<uint64_t>(model.size_gb * static_cast<double>(BYTES_PER_GB));

    // Catalog does not carry head counts. Assume a GQA ratio of 4: KV dimension
    // is then n_embd / 4 per token. This is intentionally rough; catalog
    // entries should be extended with n_head / n_head_kv for accuracy.
    const uint64_t kv_dim_per_token = static_cast<uint64_t>(n_embd) / 4;
    const uint64_t kv_per_layer_for_ctx = 2 * kv_dim_per_token * KV_ELEMENT_BYTES *
                                          static_cast<uint64_t>(n_ctx);
    result.kv_bytes = kv_per_layer_for_ctx * static_cast<uint64_t>(n_layer);

    result.compute_bytes = std::max(MIN_COMPUTE_BYTES, result.weights_bytes / 16);
    result.scratch_bytes = std::max(MIN_SCRATCH_BYTES, result.weights_bytes / 32);

    const uint64_t per_layer_weight = result.weights_bytes / static_cast<uint64_t>(n_layer);
    result.layers.reserve(n_layer);
    for (int32_t i = 0; i < n_layer; ++i) {
        model_layer_memory layer{};
        layer.layer_index          = i;
        layer.weight_bytes         = per_layer_weight;
        layer.kv_bytes_per_token   = kv_dim_per_token * 2 * KV_ELEMENT_BYTES;
        result.layers.push_back(layer);
    }

    return result;
}

model_memory_requirements estimate_model_memory_from_manifest(
        const model_manifest & manifest,
        const int32_t          n_ctx) {
    model_memory_requirements result{};
    if (manifest.empty()) {
        return result;
    }

    result.model_id = manifest.architecture.empty() ? "unknown" : manifest.architecture;
    result.n_layer  = static_cast<int32_t>(manifest.n_layer);
    result.n_embd   = static_cast<int32_t>(manifest.n_embd);
    result.n_ctx    = n_ctx;

    const int32_t n_layer = std::max(1, result.n_layer);
    const int32_t n_embd  = std::max(1, result.n_embd);

    uint64_t weights = 0;
    for (const auto & layer : manifest.layers) {
        weights += layer.size_bytes;
    }
    for (const auto & t : manifest.tensors) {
        if (t.role == tensor_role::embedding ||
                t.role == tensor_role::output_norm ||
                t.role == tensor_role::lm_head) {
            weights += t.size_bytes;
        }
    }
    if (weights == 0 && manifest.tensor_data_offset > 0) {
        weights = manifest.tensor_data_offset;
    }
    result.weights_bytes = weights;

    const uint64_t kv_dim_per_token = static_cast<uint64_t>(n_embd) / 4;
    const uint64_t kv_per_layer_for_ctx = 2 * kv_dim_per_token * KV_ELEMENT_BYTES *
                                          static_cast<uint64_t>(n_ctx);
    result.kv_bytes = kv_per_layer_for_ctx * static_cast<uint64_t>(n_layer);

    result.compute_bytes = std::max(MIN_COMPUTE_BYTES, result.weights_bytes / 16);
    result.scratch_bytes = std::max(MIN_SCRATCH_BYTES, result.weights_bytes / 32);

    result.layers.reserve(n_layer);
    for (int32_t i = 0; i < n_layer; ++i) {
        model_layer_memory layer{};
        layer.layer_index = i;
        if (i < static_cast<int32_t>(manifest.layers.size())) {
            layer.weight_bytes = manifest.layers[static_cast<size_t>(i)].size_bytes;
        } else if (n_layer > 0) {
            layer.weight_bytes = result.weights_bytes / static_cast<uint64_t>(n_layer);
        }
        layer.kv_bytes_per_token = kv_dim_per_token * 2 * KV_ELEMENT_BYTES;
        result.layers.push_back(layer);
    }

    return result;
}
