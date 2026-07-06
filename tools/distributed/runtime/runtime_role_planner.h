#pragma once

#include "runtime_graph.h"

#include "runtime_cost_model.h"
#include "runtime_descriptor.h"
#include "runtime_role_descriptor.h"

#include "dist_common.h"
#include "layer_planner.h"

#include <map>
#include <string>
#include <vector>

struct runtime_role_planner_result {
    bool           success = false;
    std::string    error;
    runtime_graph  graph;
};

// Task 11.5 — assign runtime roles independently from layer placement.
// Pipeline stages mirror layer assignments (symmetric). Services are placed by cost model.
runtime_role_planner_result dist_plan_runtime_graph(
        const runtime_descriptor & desc,
        const model_memory_requirements & mem,
        const std::vector<dist_layer_assignment> & layer_assignments,
        const std::map<std::string, dist_node_info> & nodes);

runtime_role_planner_result dist_plan_runtime_graph(
        const std::string & model_id,
        int32_t             n_layers,
        const model_memory_requirements & mem,
        const std::vector<dist_layer_assignment> & layer_assignments,
        const std::map<std::string, dist_node_info> & nodes);

// Pick lowest-cost node for a service role with memory fit constraint.
std::string dist_pick_node_for_runtime_role(
        runtime_role role,
        const runtime_role_descriptor & desc,
        const std::vector<runtime_planner_node> & candidates,
        int32_t layer_count = 0);
