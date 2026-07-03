#pragma once

#include "semantic_roles.h"

#include <cstdint>
#include <string>
#include <vector>

enum class runtime_role : uint32_t;

enum class worker_role {
    tokenizer,
    embedding,
    pipeline_stage,
    output_head,
    sampler,
    entry,
    middle,
    final,
    full,
};

std::string worker_role_to_string(worker_role role);
worker_role worker_role_from_runtime_role(runtime_role role);

struct worker_requirement {
    worker_role              role         = worker_role::entry;
    int32_t                  layer_start  = 0;
    int32_t                  layer_end    = 0;
    std::vector<std::string> required_blobs;
};

struct worker_materialize_plan {
    worker_role              role = worker_role::entry;
    int32_t                  layer_start = 0;
    int32_t                  layer_end   = 0;
    std::vector<std::string> required_blobs;
};

worker_materialize_plan materialize_plan_for_worker(
        const std::vector<worker_requirement> & requirements,
        worker_role role,
        int32_t layer_start,
        int32_t layer_end);
