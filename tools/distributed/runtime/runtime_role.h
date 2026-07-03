#pragma once

#include "nlohmann/json.hpp"

#include <cstdint>
#include <string>

// Task 11 — runtime roles (distinct from legacy dist_node_role entry/middle/final).

enum class runtime_role : uint32_t {
    unassigned     = 0,
    tokenizer      = 1,
    embedding      = 2,
    pipeline_stage = 3,
    output_head    = 4,
    sampler        = 5,
};

std::string runtime_role_name(runtime_role role);
runtime_role  runtime_role_from_string(const std::string & name);
bool runtime_role_is_service(runtime_role role);
bool runtime_role_is_pipeline(runtime_role role);

nlohmann::json runtime_role_to_json(runtime_role role);
