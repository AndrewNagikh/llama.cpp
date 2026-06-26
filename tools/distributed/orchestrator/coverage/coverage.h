#pragma once

#include "layout_planner/layout_planner.h"

#include "nlohmann/json.hpp"

#include <chrono>
#include <cstdint>
#include <set>
#include <string>
#include <vector>

// ---------------------------------------------------------------------------
// Cluster Coverage & Reconciliation - Task 9.5
//
// Compares Desired Layout (planner) with Actual State (node reports).
// Does not download models or modify files.
// ---------------------------------------------------------------------------

enum class install_state {
    missing,
    downloading,
    verifying,
    ready,
    corrupted
};

std::string install_state_to_string(install_state s);
install_state install_state_from_string(const std::string & s);

struct installed_layer {
    int32_t     layer_index = -1;
    std::string node_id;
    std::string device;
    uint64_t    size_bytes  = 0;
    std::string checksum;
    install_state state     = install_state::missing;
    std::chrono::system_clock::time_point updated_at{};
    // Semantic blob tensor (Task 9.8.3); empty for transformer layers.
    std::string blob_id;
    std::string tensor_name;

    nlohmann::json to_json() const;
    static installed_layer from_json(const nlohmann::json & j);
};

struct actual_model_layout {
    std::string model_id;
    std::vector<installed_layer> layers;

    nlohmann::json to_json() const;
    static actual_model_layout from_json(const nlohmann::json & j);
};

enum class coverage_state {
    empty,
    partial,
    ready,
    degraded
};

std::string coverage_state_to_string(coverage_state s);
coverage_state coverage_state_from_string(const std::string & s);

struct coverage_report {
    std::string model_id;
    coverage_state state = coverage_state::empty;

    int total_layers    = 0;
    int ready_layers    = 0;
    int missing_layers  = 0;
    int corrupted_layers = 0;

    std::vector<int32_t> missing;
    std::vector<int32_t> corrupted;

    nlohmann::json to_json() const;
    static coverage_report from_json(const nlohmann::json & j);
};

struct reconciliation_result {
    std::string model_id;
    coverage_state state = coverage_state::empty;
    std::vector<int32_t> missing;
    std::vector<int32_t> corrupted;

    nlohmann::json to_json() const;
    static reconciliation_result from_json(const nlohmann::json & j);
};

// Compute coverage by comparing desired placements with actual installed layers.
coverage_report compute_coverage(
        const desired_model_layout & desired,
        const actual_model_layout & actual,
        const std::set<std::string> & online_nodes = {});

// Build reconciliation action list (missing / corrupted layer indices).
reconciliation_result reconcile_layers(
        const desired_model_layout & desired,
        const actual_model_layout & actual,
        const std::set<std::string> & online_nodes = {});

// Merge node layer reports into a single actual layout.
actual_model_layout merge_node_layer_reports(
        const std::string & model_id,
        const std::vector<actual_model_layout> & node_reports);

// Parse a node GET /installed-layers response body.
actual_model_layout actual_layout_from_node_response(
        const std::string & node_id,
        const nlohmann::json & body);

// Validation helpers for tests.
bool coverage_has_no_extra_layers(
        const desired_model_layout & desired,
        const actual_model_layout & actual,
        std::string & error);
