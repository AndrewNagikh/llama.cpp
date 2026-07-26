#pragma once

#include "manifest_builder/manifest_builder.h"
#include "dist_common.h"
#include "memory_estimator.h"

#include "nlohmann/json.hpp"

#include <cstdint>
#include <map>
#include <optional>
#include <string>
#include <vector>

// ---------------------------------------------------------------------------
// Desired Cluster Layout Planner - Task 9.4
//
// Computes per-layer placement (desired state) from a Model Manifest and
// registered node capabilities. Does not download models or touch runtime.
// ---------------------------------------------------------------------------

struct layer_placement {
    int32_t     layer_index = -1;
    std::string node_id;
    std::string device;       // cpu / cuda / metal
    uint64_t    size_bytes  = 0;
    bool        required    = true;

    nlohmann::json to_json() const;
    static layer_placement from_json(const nlohmann::json & j);
};

struct desired_model_layout {
    std::string model_id;
    std::vector<layer_placement> placements;

    uint64_t total_weight_bytes    = 0;
    uint64_t total_required_memory = 0;
    bool     fits_cluster          = false;

    std::vector<std::string> warnings;

    nlohmann::json to_json() const;
    static desired_model_layout from_json(const nlohmann::json & j);
};

struct model_layout {
    desired_model_layout desired;

    nlohmann::json to_json() const;
    static model_layout from_json(const nlohmann::json & j);
};

struct layout_build_result {
    bool                 success = false;
    desired_model_layout layout;
    std::string          error;
};

// Node input for layout planning (subset of dist_node_info).
struct layout_node_input {
    std::string node_id;
    double      score = 1.0;
    std::string backend;
    bool        has_gpu = false;
    uint64_t    cpu_budget_bytes = 0;
    uint64_t    gpu_budget_bytes = 0;
    // Measured decode throughput (tokens/s) from the node's registration
    // benchmark. Drives layer counts when available for every node (Task
    // 21.4); 0 means "not measured", which falls the whole cluster back to
    // the score-proportional split.
    double      decode_tps = 0.0;
};

layout_node_input layout_node_from_dist(const dist_node_info & node);

// Convert manifest layer descriptors into memory requirements for budgeting.
model_memory_requirements memory_requirements_from_manifest(
        const model_manifest & manifest,
        int32_t n_ctx = 0);

// Relayout hysteresis: how much better a node's score must be before it is
// allowed to take a more important pipeline role away from whoever holds it
// in the previous layout. Node score is derived from live measurements
// (decode/prefill throughput, memory), so it drifts under unrelated load --
// and every role reorder moves that model's layers between nodes, orphaning
// the ones left behind (see the Task 21.1 revert in KNOWN_ISSUES.md for what
// that costs in practice: five models in DEGRADED coverage at once).
// Deliberately 2x the 5% score-delta that /register uses to notice a cluster
// change at all: we look at a 5% drift, but only act on a 10% one.
constexpr double LAYOUT_ROLE_HYSTERESIS_PERCENT = 10.0;

// Same idea one level down, for layer COUNTS (Task 21.4/21.5). Changing how
// many layers a node holds means physically re-syncing weights between nodes,
// which is far more expensive than swapping which node is entry -- so the bar
// is higher. 20% is borrowed directly from Petals, which only rebalances when
// predicted total-throughput gain clears p=20%, explicitly to trade efficiency
// against cache-invalidation cost (arXiv:2312.08361).
constexpr double LAYOUT_COUNT_HYSTERESIS_PERCENT = 20.0;

// Build desired per-layer layout from manifest + cluster nodes.
//
// `previous`, when non-null and non-empty, enables role hysteresis: the
// previous layout's node ordering is kept unless some node's score beats the
// node it would displace by more than LAYOUT_ROLE_HYSTERESIS_PERCENT. Pass
// nullptr for a fresh, unconstrained layout.
layout_build_result build_desired_layout(
        const std::string & model_id,
        const model_manifest & manifest,
        const std::vector<layout_node_input> & nodes,
        int32_t n_ctx = 0,
        const desired_model_layout * previous = nullptr);

// Validation helpers (used by tests and planner).
bool validate_desired_layout(
        const desired_model_layout & layout,
        const model_manifest & manifest,
        std::string & error);

bool layout_has_full_coverage(const desired_model_layout & layout, int32_t n_layer);
bool layout_has_no_overlap(const desired_model_layout & layout);

bool layouts_placement_equal(
        const desired_model_layout & a,
        const desired_model_layout & b);

std::string layout_normalize_device(const layout_node_input & node);
