#pragma once

#include "layout_estimator/layout_estimator.h"
#include "layout_planner/layout_planner.h"
#include "manifest_builder/manifest_builder.h"
#include "dist_common.h"

#include "nlohmann/json.hpp"

#include <cstdint>
#include <map>
#include <string>
#include <vector>

// Cluster Optimizer — Task 9.8.
// Decides whether to rebalance layout; does not download or synchronize data.

enum class node_role {
    active,
    storage_only,
    standby
};

std::string node_role_to_string(node_role role);
node_role   node_role_from_string(const std::string & s);

struct optimizer_node {
    std::string node_id;
    double      score       = 0.0;
    double      decode_tps  = 0.0;
    double      prefill_tps = 0.0;
    uint64_t    free_ram    = 0;
    uint64_t    free_vram   = 0;
    node_role   role        = node_role::active;
    bool        online      = true;

    nlohmann::json to_json() const;
};

struct optimizer_policy {
    double  min_decode_improvement_percent  = 5.0;
    double  min_prefill_improvement_percent = 5.0;
    uint64_t max_rebalance_cost_bytes       = UINT64_MAX;
    bool    allow_storage_only_nodes        = true;
    bool    prefer_gpu                      = true;
    double  storage_only_score_ratio          = 0.25;
};

enum class optimizer_decision {
    keep,
    rebalance,
    storage_only_assign
};

std::string optimizer_decision_to_string(optimizer_decision decision);

struct optimization_result {
    std::string model_id;
    optimizer_decision decision = optimizer_decision::keep;

    layout_estimate current_estimate;
    layout_estimate candidate_estimate;

    desired_model_layout current_layout;
    desired_model_layout candidate_layout;

    std::map<std::string, optimizer_node> nodes;
    std::map<std::string, node_role>      node_roles;

    std::string reason;
    bool        better = false;

    nlohmann::json to_json() const;
};

optimizer_node optimizer_node_from_dist(const dist_node_info & node);

std::map<std::string, node_role> classify_node_roles(
        const std::vector<optimizer_node> & nodes,
        const optimizer_policy & policy,
        double cluster_median_score);

optimization_result run_cluster_optimization(
        const std::string & model_id,
        const model_manifest & manifest,
        const desired_model_layout * current_layout,
        const std::vector<dist_node_info> & cluster_nodes,
        const optimizer_policy & policy = {},
        int32_t n_ctx = 0);
