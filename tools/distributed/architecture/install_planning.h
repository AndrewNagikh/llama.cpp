#pragma once

#include "architecture/semantic_runtime_descriptor.h"
#include "architecture_descriptor.h"
#include "orchestrator/install_planner/install_planner.h"
#include "runtime/runtime_install_planning.h"

#include <set>
#include <string>
#include <vector>

inline std::string blob_install_key(
        const std::string & node_id,
        const std::string & blob_id,
        const std::string & tensor_name) {
    return node_id + ":" + blob_id + ":" + tensor_name;
}

void add_semantic_blob_downloads(
        std::vector<install_operation> & operations,
        std::set<std::string> & seen,
        const semantic_runtime_descriptor & rt,
        const std::string & entry_node,
        const std::string & final_node,
        const std::vector<std::string> & all_nodes,
        const std::string & source_url,
        const actual_model_layout & actual,
        const std::set<std::string> & ready_blobs = {});

void add_semantic_blob_downloads(
        std::vector<install_operation> & operations,
        std::set<std::string> & seen,
        const semantic_runtime_descriptor & rt,
        const runtime_install_node_map & runtime_nodes,
        const std::string & source_url,
        const actual_model_layout & actual,
        const std::set<std::string> & ready_blobs = {});

void add_layer_blob_downloads(
        std::vector<install_operation> & operations,
        std::set<std::string> & seen,
        const semantic_runtime_descriptor & rt,
        const desired_model_layout & desired,
        const actual_model_layout & actual,
        const coverage_report & coverage,
        const std::string & source_url);
