#pragma once

#include "runtime_role.h"

#include "nlohmann/json.hpp"

#include <cstdint>
#include <string>

// Capability and cost hints for a runtime role on a candidate node.

struct runtime_role_descriptor {
    runtime_role role = runtime_role::unassigned;

    bool supports_tokenizer  = false;
    bool supports_embedding  = false;
    bool supports_sampling   = false;
    bool prefers_gpu         = false;

    uint64_t required_memory_bytes  = 0;
    uint64_t estimated_compute_bytes = 0;

    nlohmann::json to_json() const;
};

runtime_role_descriptor default_descriptor_for_role(
        runtime_role role,
        uint64_t     model_weight_bytes = 0);
