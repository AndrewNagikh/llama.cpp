#pragma once

#include "memory_estimator.h"

#include <string>
#include <vector>

struct dist_layer_assignment {
    std::string node_id;
    int         layer_start = 0;
    int         layer_end   = 0;
    double      score       = 0.0;
    std::string device_hint;   // "gpu" or "cpu"
};

// Score-only input. Kept for backward compatibility with Task 6 tests/tools.
struct dist_planner_node {
    std::string node_id;
    double      score = 1.0;
};

// Memory-aware input (Task 8). Budgets are independent: a GPU node should
// not be treated as having RAM+VRAM as one pool.
struct dist_planner_node_resources {
    std::string node_id;
    double      score = 1.0;
    std::string backend;

    uint64_t cpu_budget_bytes = 0;   // free RAM
    uint64_t gpu_budget_bytes = 0;   // free VRAM (0 if no GPU)
    bool     has_gpu          = false;
};

struct dist_planner_result {
    bool success = false;
    std::string error;
    std::vector<dist_layer_assignment> assignments;
};

// Proportional layer split by node score. Sorted by score descending.
// Covers [0, n_layers) exactly with no gaps or overlaps.
std::vector<dist_layer_assignment> dist_plan_layers(
        int n_layers,
        const std::vector<dist_planner_node> & nodes);

// Memory-aware, per-layer planner (Task 8).
// Assigns contiguous, gapless layer ranges while respecting each node's primary
// execution budget (GPU VRAM if present, otherwise CPU RAM). Models are not
// guaranteed to use every registered node: a node that cannot hold even one
// layer is left out.
dist_planner_result dist_plan_layers_memory_aware(
        const model_memory_requirements & mem,
        const std::vector<dist_planner_node_resources> & nodes);
