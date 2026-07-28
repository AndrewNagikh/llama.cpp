#pragma once

#include "layout_planner/layout_planner.h"

#include "nlohmann/json.hpp"

#include <chrono>
#include <cstdint>
#include <map>
#include <optional>
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
    degraded,

    // Every layer we can see is fine, but a node holding some of them is not
    // reachable. Distinct from `degraded` on purpose: nothing is damaged and
    // nothing needs downloading -- the layers are on a disk that is currently
    // switched off, and they come back with the machine.
    //
    // Before 2026-07-28 this case was reported as `degraded` with the absent
    // node's layers counted as missing, so a closed laptop was indistinguishable
    // from data loss. That reading is what prompted manual "repairs" of models
    // that were never broken (Task 24, principle 2: offline is not lost).
    unavailable
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

    // Placed on a node that is not reachable right now. Deliberately not part
    // of `missing_layers`: consumers treat that count as "must be downloaded",
    // and these must not be.
    int unavailable_layers = 0;

    std::vector<int32_t> missing;
    std::vector<int32_t> corrupted;
    std::vector<int32_t> unavailable;

    // Which machines those unavailable layers are on. Carried so a caller can
    // say "start node-c" instead of "some node is down" -- naming it is the
    // difference between an actionable message and a shrug.
    std::vector<std::string> unavailable_nodes;

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

// Expected tensor count per transformer layer (from manifest / semantic blobs).
using layer_tensor_expectations = std::map<int32_t, int>;

// Compute coverage by comparing desired placements with actual installed layers.
coverage_report compute_coverage(
        const desired_model_layout & desired,
        const actual_model_layout & actual,
        const std::set<std::string> & online_nodes = {},
        const layer_tensor_expectations * layer_tensors = nullptr);

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

// Infer committed layout from installed transformer layers (bootstrap after restart).
std::optional<desired_model_layout> desired_layout_from_actual(
        const std::string & model_id,
        const actual_model_layout & actual,
        int32_t n_layer);
