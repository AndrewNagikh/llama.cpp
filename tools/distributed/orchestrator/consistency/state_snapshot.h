#pragma once

#include "coverage/coverage.h"
#include "coverage/runtime_coverage.h"
#include "install_planner/install_planner.h"
#include "layout_planner/layout_planner.h"
#include "manifest_builder/manifest_builder.h"

#include "nlohmann/json.hpp"

#include <string>

// Task 9.9 — point-in-time cluster state snapshot after install steps.

struct cluster_state_snapshot {
    std::string model_id;
    std::string captured_at;

    desired_model_layout desired;
    actual_model_layout  actual;
    coverage_report      coverage;
    runtime_coverage_report runtime;
    install_plan         plan;

    nlohmann::json layer_store_summary = nlohmann::json::object();

    nlohmann::json to_json() const;
    static cluster_state_snapshot from_json(const nlohmann::json & j);
};

cluster_state_snapshot capture_cluster_snapshot(
        const std::string & model_id,
        const desired_model_layout & desired,
        const actual_model_layout & actual,
        const coverage_report & coverage,
        const runtime_coverage_report & runtime,
        const install_plan & plan,
        const nlohmann::json & layer_store_summary = {});

nlohmann::json diff_snapshots(
        const cluster_state_snapshot & before,
        const cluster_state_snapshot & after);
