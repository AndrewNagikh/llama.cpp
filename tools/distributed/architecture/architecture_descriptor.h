#pragma once

#include "semantic_blob.h"
#include "worker_requirement.h"

#include "orchestrator/manifest_builder/manifest_builder.h"

#include <string>
#include <vector>

struct architecture_descriptor {
    std::string architecture;
    std::string family;

    bool tied_embeddings  = false;
    bool separate_lm_head = false;
    bool is_moe           = false;

    std::vector<semantic_blob>       blobs;
    std::vector<worker_requirement> worker_requirements;

    bool empty() const { return architecture.empty() && blobs.empty(); }
};

architecture_descriptor build_architecture_descriptor(const model_manifest & manifest);

std::vector<std::string> nodes_for_blob_deploy(
        blob_deploy_target deploy,
        const std::string & entry_node,
        const std::string & final_node,
        const std::vector<std::string> & all_nodes);
