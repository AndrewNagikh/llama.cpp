#include "runtime_graph.h"

#include <algorithm>

using json = nlohmann::json;

nlohmann::json runtime_service_endpoint::to_json() const {
    return {
        { "scheme", scheme.empty() ? "http" : scheme },
        { "host", host },
        { "port", port },
    };
}

runtime_service_endpoint runtime_service_endpoint::from_json(const nlohmann::json & j) {
    runtime_service_endpoint e{};
    e.scheme = j.value("scheme", "http");
    e.host   = j.value("host", "");
    e.port   = j.value("port", 0);
    return e;
}

nlohmann::json runtime_role_assignment::to_json() const {
    runtime_service_endpoint effective_endpoint = endpoint;
    if (effective_endpoint.host.empty()) {
        effective_endpoint.host = host;
    }
    if (effective_endpoint.port <= 0) {
        effective_endpoint.port = http_port;
    }
    json j = {
        { "role", runtime_role_name(role) },
        { "node_id", node_id },
        { "host", host },
        { "http_port", http_port },
        { "endpoint", effective_endpoint.to_json() },
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
    if (j.contains("endpoint") && j["endpoint"].is_object()) {
        a.endpoint = runtime_service_endpoint::from_json(j["endpoint"]);
    } else {
        a.endpoint.host = a.host;
        a.endpoint.port = a.http_port;
    }
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
