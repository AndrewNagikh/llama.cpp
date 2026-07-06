#pragma once

#include "architecture/semantic_runtime_descriptor.h"
#include "coverage.h"
#include "layout_planner/layout_planner.h"
#include "runtime/runtime_install_planning.h"

#include <set>
#include <string>
#include <vector>

// Task 9.9 — coverage beyond layer indices.

struct runtime_coverage_report {
    coverage_report layer_coverage;
    coverage_state  storage_state = coverage_state::empty;
    coverage_state  runtime_state = coverage_state::empty;

    bool storage_ready() const { return storage_state == coverage_state::ready; }
    bool runtime_ready() const { return runtime_state == coverage_state::ready; }
    bool fully_ready() const {
        return layer_coverage.state == coverage_state::ready && runtime_ready();
    }

    std::vector<std::string> missing_blobs;
    std::vector<std::string> misplaced_blobs;

    nlohmann::json to_json() const;
};

runtime_coverage_report compute_runtime_coverage(
        const semantic_runtime_descriptor & rt,
        const desired_model_layout & desired,
        const actual_model_layout & actual,
        const std::set<std::string> & online_nodes = {},
        const runtime_install_node_map * runtime_nodes = nullptr);
