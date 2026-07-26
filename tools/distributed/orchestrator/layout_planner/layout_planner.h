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
};

layout_node_input layout_node_from_dist(const dist_node_info & node);

// Convert manifest layer descriptors into memory requirements for budgeting.
model_memory_requirements memory_requirements_from_manifest(
        const model_manifest & manifest,
        int32_t n_ctx = 0);

// Build desired per-layer layout from manifest + cluster nodes.
layout_build_result build_desired_layout(
        const std::string & model_id,
        const model_manifest & manifest,
        const std::vector<layout_node_input> & nodes,
        int32_t n_ctx = 0);

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
