#include "runtime_graph.h"

#include <algorithm>

using json = nlohmann::json;

nlohmann::json runtime_role_assignment::to_json() const {
    json j = {
        { "role", runtime_role_name(role) },
        { "node_id", node_id },
        { "host", host },
        { "http_port", http_port },
        { "layer_start", layer_start },
        { "layer_end", layer_end },
        { "stage_index", stage_index },
        { "score", score },
    };
    if (ctrl_port > 0) {
        j["ctrl_port"] = ctrl_port;
    }
    if (peer_port > 0) {
        j["peer_port"] = peer_port;
    }
    return j;
}

runtime_role_assignment runtime_role_assignment::from_json(const nlohmann::json & j) {
    runtime_role_assignment a{};
    a.role        = runtime_role_from_string(j.value("role", ""));
    a.node_id     = j.value("node_id", "");
    a.host        = j.value("host", "");
    a.http_port   = j.value("http_port", 0);
    a.layer_start = j.value("layer_start", 0);
    a.layer_end   = j.value("layer_end", 0);
    a.stage_index = j.value("stage_index", 0);
    a.ctrl_port   = j.value("ctrl_port", 0);
    a.peer_port   = j.value("peer_port", 0);
    a.score       = j.value("score", 0.0);
    return a;
}

const runtime_role_assignment * runtime_graph::find_role(const runtime_role role) const {
    for (const auto & a : assignments) {
        if (a.role == role) {
            return &a;
        }
    }
    return nullptr;
}

std::vector<const runtime_role_assignment *> runtime_graph::pipeline_stages() const {
    std::vector<const runtime_role_assignment *> out;
    for (const auto & a : assignments) {
        if (a.role == runtime_role::pipeline_stage) {
            out.push_back(&a);
        }
    }
    std::sort(out.begin(), out.end(), [](const runtime_role_assignment * x, const runtime_role_assignment * y) {
        if (x->layer_start != y->layer_start) {
            return x->layer_start < y->layer_start;
        }
        return x->stage_index < y->stage_index;
    });
    return out;
}

nlohmann::json runtime_graph::to_json() const {
    json roles = json::array();
    for (const auto & a : assignments) {
        roles.push_back(a.to_json());
    }
    return {
        { "model_id", model_id },
        { "n_layers", n_layers },
        { "roles", roles },
    };
}
