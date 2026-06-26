#pragma once

#include "coverage/coverage.h"
#include "layout_planner/layout_planner.h"
#include "manifest_builder/manifest_builder.h"
#include "node_agent/layer_store/layer_special.h"

#include "nlohmann/json.hpp"

#include <cstdint>
#include <string>
#include <vector>

// ---------------------------------------------------------------------------
// Install Planner - Task 9.6
//
// Converts Desired + Actual + Coverage into a deterministic Install Plan.
// Does not perform HTTP, downloads, or shard creation.
// ---------------------------------------------------------------------------

enum class install_action {
    download,
    verify,
    delete_op,
    repair
};

std::string install_action_to_string(install_action action);
install_action install_action_from_string(const std::string & s);

struct layer_byte_range {
    uint64_t    offset     = 0;
    uint64_t    length     = 0;
    uint64_t    size_bytes = 0;
    std::string checksum;
};

struct download_operation {
    int32_t     layer_index   = -1;
    std::string node_id;
    uint64_t    tensor_offset = 0;
    uint64_t    tensor_length = 0;
    std::string source_url;
    std::string checksum;

    nlohmann::json to_json() const;
    static download_operation from_json(const nlohmann::json & j);
};

struct install_operation {
    install_action      action = install_action::download;
    std::string         node_id;
    int32_t             layer_index = -1;
    download_operation  download;

    nlohmann::json to_json() const;
    static install_operation from_json(const nlohmann::json & j);
};

struct install_plan_group {
    std::string              node_id;
    install_action           action = install_action::download;
    std::vector<int32_t>     layers;
    uint64_t                 download_bytes = 0;

    nlohmann::json to_json() const;
    static install_plan_group from_json(const nlohmann::json & j);
};

struct install_plan {
    std::string model_id;
    std::vector<install_operation> operations;
    std::vector<install_plan_group>  groups;

    uint64_t total_download_bytes = 0;
    int      operation_count      = 0;

    nlohmann::json to_json() const;
    static install_plan from_json(const nlohmann::json & j);
};

struct install_plan_build_result {
    bool         success = false;
    install_plan plan;
    std::string  error;
};

// Compute byte range for a layer from manifest tensor descriptors.
layer_byte_range manifest_layer_byte_range(
        const model_manifest & manifest,
        int32_t layer_index);

layer_byte_range manifest_role_byte_range(
        const model_manifest & manifest,
        tensor_role role);

// All global tensors before the first transformer layer (rope, embedding, norms).
layer_byte_range manifest_global_preamble_range(const model_manifest & manifest);

// Build install plan from registry state inputs.
install_plan_build_result build_install_plan(
        const model_manifest & manifest,
        const desired_model_layout & desired,
        const actual_model_layout & actual,
        const coverage_report & coverage,
        const std::string & source_url = "");

// Group consecutive same-node operations (sorted by node, layer).
std::vector<install_plan_group> group_install_operations(
        const std::vector<install_operation> & operations);

// Validation helpers for tests.
bool validate_install_plan(
        const install_plan & plan,
        const desired_model_layout & desired,
        const coverage_report & coverage,
        std::string & error);

bool install_plan_has_grouped_layers(
        const install_plan & plan,
        const std::string & node_id,
        install_action action,
        const std::vector<int32_t> & expected_layers);
