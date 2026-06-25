#pragma once

#include "dist_common.h"
#include "model_catalog.h"

#include "nlohmann/json.hpp"

#include <cstdint>
#include <string>
#include <vector>

// Memory requirements for a single transformer layer.
// This is an intentionally architecture-neutral descriptor: Task 9 will plug in
// real per-layer sizes from a layer-sharded catalog while the planner itself
// stays unchanged.
struct model_layer_memory {
    int32_t  layer_index  = 0;
    uint64_t weight_bytes = 0;

    // Per-token KV increment produced by this layer. Carrying this per-layer
    // keeps the structures ready for heterogeneous layer sizes (e.g. MLA,
    // SWA, sparse layers) in Task 11.
    uint64_t kv_bytes_per_token = 0;
};

struct model_memory_requirements {
    std::string model_id;
    int32_t     n_layer = 0;
    int32_t     n_embd  = 0;
    int32_t     n_ctx   = 4096;

    // One entry per layer. Used by the planner for per-layer assignment.
    std::vector<model_layer_memory> layers;

    // Aggregated totals.
    uint64_t weights_bytes = 0;
    uint64_t kv_bytes      = 0;
    uint64_t compute_bytes = 0;
    uint64_t scratch_bytes = 0;

    uint64_t total_bytes() const {
        return weights_bytes + kv_bytes + compute_bytes + scratch_bytes;
    }

    double weights_gb() const;
    double kv_gb() const;
    double compute_gb() const;
    double scratch_gb() const;
    double total_gb() const;

    bool valid() const {
        return n_layer > 0 && weights_bytes > 0;
    }
};

// Feasibility result for the whole cluster.
// Note: available_gb is the sum of each node's primary execution budget
// (GPU VRAM if present, otherwise CPU RAM). RAM and VRAM are never summed.
struct cluster_memory_fits_result {
    bool fits = false;
    double required_gb = 0.0;
    double available_gb = 0.0;
    double missing_gb = 0.0;
    std::vector<std::string> warnings;

    nlohmann::json to_json() const;
};

cluster_memory_fits_result dist_check_cluster_memory_fit(
        const model_memory_requirements & mem,
        const std::vector<dist_node_info> & nodes);

// Estimate memory from a local GGUF file. Reads real model metadata.
model_memory_requirements estimate_model_memory(
        const std::string & gguf_path,
        int32_t             n_ctx = 4096);

// Estimate memory from a catalog entry. Used when the orchestrator has no local GGUF.
model_memory_requirements estimate_model_memory_from_catalog(
        const model_info & model,
        int32_t            n_ctx = 4096);
