#pragma once

#include "coverage/coverage.h"
#include "coverage/runtime_coverage.h"
#include "install_planner/install_planner.h"
#include "manifest_builder/manifest_builder.h"
#include "model_registry.h"

#include "state_snapshot.h"

#include "nlohmann/json.hpp"

#include <set>
#include <string>
#include <vector>

// Task 9.9 — cross-check Registry, Coverage, Layer Store, Install Plan.

struct install_plan_diff {
    int added_ops   = 0;
    int removed_ops = 0;
    std::vector<nlohmann::json> operations;

    nlohmann::json to_json() const;
};

struct consistency_check_result {
    bool consistent  = false;
    bool idempotent  = false;
    bool coverage_ready = false;

    install_plan rebuilt_plan;
    install_plan_diff plan_diff;

    std::vector<std::string> issues;
    cluster_state_snapshot snapshot;

    nlohmann::json to_json() const;
};

install_plan_diff diff_install_plan(
        const install_plan & baseline,
        const install_plan & candidate);

consistency_check_result check_cluster_consistency(
        const dist_model_record & record,
        const actual_model_layout & actual,
        const std::set<std::string> & online_nodes,
        const std::string & source_url);

// Coverage must not flip READY→PARTIAL when store/registry unchanged.
bool assert_coverage_stable(
        const coverage_report & before,
        const coverage_report & after,
        bool store_unchanged,
        std::string & error);

bool coverage_fully_ready(const coverage_report & coverage);

bool install_plan_is_idempotent(const install_plan & plan);
