#include "runtime_role_planner.h"

#include "runtime_cost_model.h"
#include "runtime_descriptor.h"
#include "runtime_role_descriptor.h"

#include <algorithm>
#include <map>
#include <set>

std::string dist_pick_node_for_runtime_role(
        const runtime_role role,
        const runtime_role_descriptor & desc,
        const std::vector<runtime_planner_node> & candidates,
        const int32_t layer_count) {
    std::string best_id;
    double best_cost = 1e18;
    for (const auto & node : candidates) {
        const double cost = runtime_cost_for_role(role, node, desc, layer_count);
        if (best_id.empty() || cost < best_cost ||
                (cost == best_cost && node.node_id < best_id)) {
            best_cost = cost;
            best_id   = node.node_id;
        }
    }
    return best_id;
}

static runtime_role_descriptor role_descriptor_from_service(
        const runtime_service_descriptor & service,
        const uint64_t model_weight_bytes) {
    runtime_role_descriptor d =
            default_descriptor_for_role(service.role, model_weight_bytes);
    if (service.cost.memory_static_bytes > 0) {
        d.required_memory_bytes = service.cost.memory_static_bytes;
    }
    if (service.cost.compute_weight_prefill > 0 ||
            service.cost.compute_weight_decode > 0) {
        d.estimated_compute_bytes =
                service.cost.compute_weight_prefill +
                service.cost.compute_weight_decode;
    }
    d.supports_tokenizer = service.role == runtime_role::tokenizer;
    d.supports_embedding = service.role == runtime_role::embedding;
    d.supports_sampling  = service.role == runtime_role::sampler;
    return d;
}

runtime_role_planner_result dist_plan_runtime_graph(
        const runtime_descriptor & desc,
        const model_memory_requirements & mem,
        const std::vector<dist_layer_assignment> & layer_assignments,
        const std::map<std::string, dist_node_info> & nodes) {
    runtime_role_planner_result result{};
    result.graph.model_id = desc.model_id;
    result.graph.n_layers = mem.n_layer;

    const runtime_descriptor_validation desc_validation =
            validate_runtime_descriptor(desc);
    if (!desc_validation.ok()) {
        result.error = desc_validation.errors.empty() ?
                "invalid runtime descriptor" :
                desc_validation.errors.front();
        return result;
    }

    if (layer_assignments.empty()) {
        result.error = "empty layer assignments";
        return result;
    }
    if (nodes.empty()) {
        result.error = "no nodes";
        return result;
    }

    std::vector<runtime_planner_node> candidates;
    candidates.reserve(nodes.size());
    std::map<std::string, int32_t> layers_per_node;
    for (const auto & la : layer_assignments) {
        layers_per_node[la.node_id] += la.layer_end - la.layer_start;
    }
    const std::string first_stage_node = layer_assignments.front().node_id;
    const std::string last_stage_node  = layer_assignments.back().node_id;

    for (const auto & kv : nodes) {
        if (kv.second.online) {
            runtime_planner_node pn = runtime_planner_node_from_dist(kv.second);
            pn.pipeline_layers = layers_per_node[kv.first];
            pn.is_first_pipeline_stage = kv.first == first_stage_node;
            pn.is_last_pipeline_stage  = kv.first == last_stage_node;
            candidates.push_back(std::move(pn));
        }
    }
    if (candidates.empty()) {
        result.error = "no online nodes";
        return result;
    }

    std::sort(candidates.begin(), candidates.end(), [](const runtime_planner_node & a,
                                                       const runtime_planner_node & b) {
        return a.node_id < b.node_id;
    });

    const uint64_t weight_bytes = mem.weights_bytes > 0 ? mem.weights_bytes : 1;

    auto preferred_node_for_service = [&](runtime_role role) -> std::string {
        switch (role) {
            case runtime_role::tokenizer:
            case runtime_role::embedding:
                return first_stage_node;
            case runtime_role::output_head:
            case runtime_role::sampler:
                return last_stage_node;
            default:
                return "";
        }
    };

    auto add_service = [&](runtime_role role, const std::string & node_id) -> bool {
        const auto it = nodes.find(node_id);
        if (it == nodes.end()) {
            result.error = "unknown node for role " + runtime_role_name(role) + ": " + node_id;
            return false;
        }
        runtime_role_assignment a{};
        a.role      = role;
        a.node_id   = node_id;
        a.host      = it->second.host;
        a.http_port = it->second.http_port;
        a.endpoint.host = it->second.host;
        a.endpoint.port = it->second.http_port;
        a.score     = it->second.score;
        result.graph.assignments.push_back(std::move(a));
        return true;
    };

    for (const runtime_service_descriptor & service : desc.services) {
        if (service.role == runtime_role::pipeline_stage) {
            continue;
        }
        const runtime_role_descriptor role_desc =
                role_descriptor_from_service(service, weight_bytes);
        std::string node_id = preferred_node_for_service(service.role);
        if (node_id.empty() || nodes.find(node_id) == nodes.end() || !nodes.at(node_id).online) {
            node_id = dist_pick_node_for_runtime_role(service.role, role_desc, candidates, 0);
        }
        if (node_id.empty()) {
            result.error = "failed to assign service role: " + service.name;
            return result;
        }
        if (!add_service(service.role, node_id)) {
            return result;
        }
    }

    int stage_idx = 0;
    for (const auto & la : layer_assignments) {
        const auto it = nodes.find(la.node_id);
        if (it == nodes.end()) {
            result.error = "layer assignment references unknown node: " + la.node_id;
            return result;
        }
        const int32_t layer_count = la.layer_end - la.layer_start;
        (void) layer_count;

        runtime_role_assignment stage{};
        stage.role        = runtime_role::pipeline_stage;
        stage.node_id     = la.node_id;
        stage.host        = it->second.host;
        stage.http_port   = it->second.http_port;
        stage.endpoint.host = it->second.host;
        stage.endpoint.port = it->second.http_port;
        stage.layer_start = la.layer_start;
        stage.layer_end   = la.layer_end;
        stage.stage_index = stage_idx++;
        stage.score       = la.score;
        result.graph.assignments.push_back(std::move(stage));
    }

    result.success = true;
    return result;
}

runtime_role_planner_result dist_plan_runtime_graph(
        const std::string & model_id,
        const int32_t       n_layers,
        const model_memory_requirements & mem,
        const std::vector<dist_layer_assignment> & layer_assignments,
        const std::map<std::string, dist_node_info> & nodes) {
    runtime_descriptor desc =
            make_basic_text_generation_runtime_descriptor(model_id, "legacy.arch", n_layers);
    model_memory_requirements effective_mem = mem;
    if (effective_mem.n_layer <= 0) {
        effective_mem.n_layer = n_layers;
    }
    return dist_plan_runtime_graph(desc, effective_mem, layer_assignments, nodes);
}
