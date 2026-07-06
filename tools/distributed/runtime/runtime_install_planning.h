#pragma once

#include "runtime_graph.h"

#include "architecture/semantic_blob.h"
#include "runtime_descriptor.h"

#include <string>
#include <vector>

// Task 11.9 — map semantic blob deploy targets to runtime-graph role nodes.

struct runtime_install_node_map {
    std::string              tokenizer_node;
    std::string              embedding_node;
    std::string              output_head_node;
    std::string              first_pipeline_node;
    std::string              last_pipeline_node;
    std::vector<std::string> all_nodes;

    bool valid() const {
        return !all_nodes.empty() &&
               !tokenizer_node.empty() &&
               !embedding_node.empty() &&
               !output_head_node.empty();
    }
};

runtime_install_node_map runtime_install_node_map_from_graph(const runtime_graph & graph);

runtime_install_node_map runtime_install_node_map_legacy(
        const std::string & entry_node,
        const std::string & final_node,
        const std::vector<std::string> & pipeline_nodes);

std::vector<std::string> nodes_for_runtime_blob_deploy(
        blob_deploy_target deploy,
        tensor_semantic_role role,
        const runtime_install_node_map & map);

struct runtime_resource_install_assignment {
    std::string  resource_id;
    runtime_role service_role = runtime_role::unassigned;
    std::string  node_id;
    bool         required = true;
};

struct runtime_resource_install_plan {
    bool success = false;
    std::string error;
    std::vector<runtime_resource_install_assignment> assignments;
};

runtime_resource_install_plan build_runtime_resource_install_plan(
        const runtime_descriptor & desc,
        const runtime_graph & graph);

runtime_resource_install_plan validate_runtime_resource_install_plan(
        const runtime_descriptor & desc,
        const runtime_graph & graph,
        const runtime_resource_install_plan & plan);

bool runtime_resource_plan_has(
        const runtime_resource_install_plan & plan,
        const std::string & resource_id,
        runtime_role service_role,
        const std::string & node_id);
