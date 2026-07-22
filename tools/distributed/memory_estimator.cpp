#include "memory_estimator.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <string>

static constexpr uint64_t BYTES_PER_GB         = 1024ULL * 1024ULL * 1024ULL;
static constexpr uint64_t KV_ELEMENT_BYTES     = 2;   // fp16/bf16 KV cache cells.
static constexpr uint64_t MIN_COMPUTE_BYTES    = 256ULL * 1024ULL * 1024ULL;
static constexpr uint64_t MIN_SCRATCH_BYTES    = 128ULL * 1024ULL * 1024ULL;

static double bytes_to_gb(const uint64_t bytes) {
    return static_cast<double>(bytes) / static_cast<double>(BYTES_PER_GB);
}

// compute_bytes/scratch_bytes below are a crude weights-proportional
// heuristic with no architecture awareness at all -- confirmed empirically
// 2026-07-23: it silently OK'd a Qwen3-30B-A3B (qwen3moe) layout that then
// OOM-crashed node-b during graph_reserve with no diagnostic (the crash
// happens inside ggml's real allocation, which our pre-flight check never
// sees). MoE's per-token top-k expert gather/scatter needs materially more
// intermediate/routing buffer than a dense model's same weight count, and
// the manifest's real per-layer weight_bytes (correctly larger for MoE
// layers, since it reads actual GGUF tensor sizes) doesn't capture that --
// only the *weights* are architecture-accurate here, not the runtime
// scratch need. This multiplier is a safety-margin guess, not a measured
// model: pending calibration from an actual successful MoE run's observed
// memory use (see docs/FIRST_SHOWCASE_CRITERIA.md G1 / TASK_21 follow-up).
// Erring toward overestimating -- a layout that's too conservative just
// wastes a bit of capacity; one that's too optimistic crashes a node.
// Not static: orchestrator/layout_planner/layout_planner.cpp has its own,
// separate (duplicated) memory_requirements_from_manifest() with the same
// crude heuristic and needs the same MoE detection -- exposed via
// memory_estimator.h rather than tripling this string-matching logic.
// Unifying the two implementations outright is a real follow-up (they've
// independently drifted before, e.g. duplicated MIN_COMPUTE_BYTES/
// MIN_SCRATCH_BYTES constants), just not tonight's fix.
bool model_architecture_is_moe(const std::string & architecture) {
    std::string lower = architecture;
    std::transform(lower.begin(), lower.end(), lower.begin(),
            [](unsigned char c) { return std::tolower(c); });
    return lower.find("moe") != std::string::npos;
}

static void apply_compute_scratch_estimate(model_memory_requirements & result, bool is_moe) {
    const uint64_t compute_divisor = is_moe ? 4  : 16;
    const uint64_t scratch_divisor = is_moe ? 8  : 32;
    result.compute_bytes = std::max(MIN_COMPUTE_BYTES, result.weights_bytes / compute_divisor);
    result.scratch_bytes = std::max(MIN_SCRATCH_BYTES, result.weights_bytes / scratch_divisor);
}

double model_memory_requirements::weights_gb() const { return bytes_to_gb(weights_bytes); }
double model_memory_requirements::kv_gb()      const { return bytes_to_gb(kv_bytes); }
double model_memory_requirements::compute_gb() const { return bytes_to_gb(compute_bytes); }
double model_memory_requirements::scratch_gb() const { return bytes_to_gb(scratch_bytes); }
double model_memory_requirements::total_gb()   const { return bytes_to_gb(total_bytes()); }

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

    // Catalog entries carry no architecture string -- can't detect MoE here,
    // falls back to the dense heuristic. Registered models go through
    // estimate_model_memory_from_manifest below instead, which can.
    apply_compute_scratch_estimate(result, /*is_moe=*/false);

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

    apply_compute_scratch_estimate(result, model_architecture_is_moe(manifest.architecture));

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
