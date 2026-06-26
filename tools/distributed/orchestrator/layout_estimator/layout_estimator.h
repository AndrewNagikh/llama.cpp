#pragma once

#include "layout_planner/layout_planner.h"
#include "manifest_builder/manifest_builder.h"
#include "install_planner/install_planner.h"

#include <cstdint>
#include <map>
#include <string>

// Pipeline performance and rebalance cost estimates (Task 9.8).

struct layout_estimate {
    double estimated_decode_tps          = 0.0;
    double estimated_prefill_tps         = 0.0;
    double estimated_pipeline_latency_ms = 0.0;
    double network_cost                  = 0.0;
    uint64_t rebalance_cost_bytes        = 0;
    bool     better_than_current         = false;

    nlohmann::json to_json() const;
};

layout_estimate estimate_layout_performance(
        const desired_model_layout & layout,
        const std::map<std::string, double> & node_decode_tps,
        const std::map<std::string, double> & node_prefill_tps);

uint64_t estimate_rebalance_cost_bytes(
        const desired_model_layout & current_layout,
        const desired_model_layout & candidate_layout,
        const model_manifest & manifest);

bool layout_estimate_is_better(
        const layout_estimate & current,
        const layout_estimate & candidate,
        double min_decode_improvement_percent,
        double min_prefill_improvement_percent,
        uint64_t max_rebalance_cost_bytes);
