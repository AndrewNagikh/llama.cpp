#pragma once

#include "architecture_descriptor.h"
#include "worker_requirement.h"

#include "orchestrator/manifest_builder/manifest_builder.h"

#include <string>
#include <vector>

// Task 9.9 — single source of truth for install, materialize, coverage, workers.

struct worker_descriptor {
    worker_role              role = worker_role::entry;
    std::vector<std::string> required_blob_ids;
    std::vector<int32_t>     required_layer_indices;
};

struct semantic_runtime_descriptor {
    std::string architecture;
    std::string family;

    bool tied_embeddings  = false;
    bool separate_lm_head = false;
    bool is_moe           = false;

    std::vector<semantic_blob>       blobs;
    std::vector<worker_descriptor>   workers;

    bool empty() const { return architecture.empty() && blobs.empty(); }
};

semantic_runtime_descriptor build_semantic_runtime_descriptor(const model_manifest & manifest);

worker_materialize_plan worker_materialize_plan_for_role(
        const semantic_runtime_descriptor & rt,
        worker_role role,
        int32_t layer_start,
        int32_t layer_end);

const worker_descriptor * find_worker_descriptor(
        const semantic_runtime_descriptor & rt,
        worker_role role);
