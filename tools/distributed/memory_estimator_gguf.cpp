#include "memory_estimator.h"

#include "llama.h"

#include "ggml-backend.h"

#include <algorithm>
#include <cstdio>
#include <filesystem>
#include <string>

static constexpr uint64_t BYTES_PER_GB      = 1024ULL * 1024ULL * 1024ULL;
static constexpr uint64_t KV_ELEMENT_BYTES    = 2;
static constexpr uint64_t MIN_COMPUTE_BYTES   = 256ULL * 1024ULL * 1024ULL;
static constexpr uint64_t MIN_SCRATCH_BYTES   = 128ULL * 1024ULL * 1024ULL;

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
    result.model_id      = arch.empty() ? "unknown" : arch;
    result.n_layer       = n_layer;
    result.n_embd        = n_embd;
    result.n_ctx         = n_ctx;
    result.weights_bytes = file_size > 0 ? file_size : 0;

    const uint64_t kv_per_token_per_layer = static_cast<uint64_t>(2) *
                                            static_cast<uint64_t>(n_head_kv) *
                                            static_cast<uint64_t>(head_dim) *
                                            KV_ELEMENT_BYTES;
    const uint64_t kv_per_layer_for_ctx = kv_per_token_per_layer * static_cast<uint64_t>(n_ctx);
    result.kv_bytes = kv_per_layer_for_ctx * static_cast<uint64_t>(std::max(1, n_layer));

    result.compute_bytes = std::max(MIN_COMPUTE_BYTES, result.weights_bytes / 16);
    result.scratch_bytes = std::max(MIN_SCRATCH_BYTES, result.weights_bytes / 32);

    const uint64_t per_layer_weight = n_layer > 0 ? (result.weights_bytes / static_cast<uint64_t>(n_layer)) : 0;
    result.layers.reserve(n_layer);
    for (int32_t i = 0; i < n_layer; ++i) {
        model_layer_memory layer{};
        layer.layer_index        = i;
        layer.weight_bytes       = per_layer_weight;
        layer.kv_bytes_per_token = kv_per_token_per_layer;
        result.layers.push_back(layer);
    }

    llama_model_free(model);
    return result;
}
