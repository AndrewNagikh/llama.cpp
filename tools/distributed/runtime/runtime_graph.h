#pragma once

#include "runtime_role.h"

#include "nlohmann/json.hpp"

#include <cstdint>
#include <string>
#include <vector>

// One role binding on a cluster node.

struct runtime_role_assignment {
    runtime_role role         = runtime_role::unassigned;
    std::string  node_id;
    std::string  host;
    int          http_port    = 0;
    int32_t      layer_start  = 0;
    int32_t      layer_end    = 0;
    int          stage_index  = 0;
    int          ctrl_port    = 0;
    int          peer_port    = 0;
    double       score        = 0.0;

    nlohmann::json to_json() const;
    static runtime_role_assignment from_json(const nlohmann::json & j);
};

struct runtime_graph {
    std::string model_id;
    int32_t     n_layers = 0;
    std::vector<runtime_role_assignment> assignments;

    const runtime_role_assignment * find_role(runtime_role role) const;
    std::vector<const runtime_role_assignment *> pipeline_stages() const;
    nlohmann::json to_json() const;
};
