#pragma once

#include "architecture_descriptor.h"
#include "orchestrator/install_planner/install_planner.h"

#include <set>
#include <string>
#include <vector>

void add_semantic_blob_downloads(
        std::vector<install_operation> & operations,
        std::set<std::string> & seen,
        const architecture_descriptor & desc,
        const model_manifest & manifest,
        const std::string & entry_node,
        const std::string & final_node,
        const std::vector<std::string> & all_nodes,
        const std::string & source_url);
