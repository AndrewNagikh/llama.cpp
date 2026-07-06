#pragma once

#include "runtime_role_descriptor.h"

#include "dist_common.h"
#include "memory_estimator.h"

#include <cstdint>
#include <string>
#include <vector>

struct runtime_planner_node {
    std::string node_id;
    double      score           = 1.0;
    std::string backend;
    bool        has_gpu         = false;
    uint64_t    cpu_budget_bytes = 0;
    uint64_t    gpu_budget_bytes = 0;
    double      cpu_score       = 1.0;
    double      memory_bw_score = 1.0;
    int32_t     pipeline_layers = 0;
    bool        is_first_pipeline_stage = false;
    bool        is_last_pipeline_stage  = false;
};

runtime_planner_node runtime_planner_node_from_dist(const dist_node_info & node);

double runtime_cost_tokenizer(const runtime_planner_node & node, const runtime_role_descriptor & desc);
double runtime_cost_embedding(const runtime_planner_node & node, const runtime_role_descriptor & desc);
double runtime_cost_pipeline_stage(
        const runtime_planner_node & node,
        const runtime_role_descriptor & desc,
        int32_t layer_count);
double runtime_cost_output_head(const runtime_planner_node & node, const runtime_role_descriptor & desc);
double runtime_cost_sampler(const runtime_planner_node & node, const runtime_role_descriptor & desc);

double runtime_cost_for_role(
        runtime_role role,
        const runtime_planner_node & node,
        const runtime_role_descriptor & desc,
        int32_t layer_count = 0);

uint64_t runtime_node_budget_bytes(const runtime_planner_node & node, bool prefers_gpu);

double runtime_pipeline_service_penalty(
        const runtime_planner_node & node,
        runtime_role role);