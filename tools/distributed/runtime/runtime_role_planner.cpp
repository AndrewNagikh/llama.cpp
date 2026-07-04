#include "runtime_role_planner.h"

#include "runtime_cost_model.h"
#include "runtime_role_descriptor.h"

#include <algorithm>
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
        if (cost < best_cost) {
            best_cost = cost;
            best_id   = node.node_id;
        }
    }
    return best_id;
}

runtime_role_planner_result dist_plan_runtime_graph(
        const std::string & model_id,
        const int32_t       n_layers,
        const model_memory_requirements & mem,
        const std::vector<dist_layer_assignment> & layer_assignments,
        const std::map<std::string, dist_node_info> & nodes) {
    runtime_role_planner_result result{};
    result.graph.model_id = model_id;
    result.graph.n_layers = n_layers;

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
    for (const auto & kv : nodes) {
        if (kv.second.online) {
            candidates.push_back(runtime_planner_node_from_dist(kv.second));
        }
    }
    if (candidates.empty()) {
        result.error = "no online nodes";
        return result;
    }

    const uint64_t weight_bytes = mem.weights_bytes > 0 ? mem.weights_bytes : 1;

    const runtime_role_descriptor tok_desc = default_descriptor_for_role(runtime_role::tokenizer, weight_bytes);
    const runtime_role_descriptor emb_desc = default_descriptor_for_role(runtime_role::embedding, weight_bytes);
    const runtime_role_descriptor out_desc = default_descriptor_for_role(runtime_role::output_head, weight_bytes);
    const runtime_role_descriptor smp_desc = default_descriptor_for_role(runtime_role::sampler, weight_bytes);
    const runtime_role_descriptor stg_desc = default_descriptor_for_role(runtime_role::pipeline_stage, weight_bytes);

    const std::string first_stage_node = layer_assignments.front().node_id;
    const std::string last_stage_node  = layer_assignments.back().node_id;

    // Tokenizer/embedding/output colocate with first/last pipeline stage until
    // dedicated blob sync and hidden/logits paths (11.3–11.4) are wired.
    (void) tok_desc;
    (void) emb_desc;
    (void) out_desc;
    const std::string tok_node         = first_stage_node;
    const std::string emb_node         = first_stage_node;
    const std::string out_node         = last_stage_node;

    const std::string smp_node = dist_pick_node_for_runtime_role(runtime_role::sampler, smp_desc, candidates);

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
        a.score     = it->second.score;
        result.graph.assignments.push_back(std::move(a));
        return true;
    };

    if (tok_node.empty() || emb_node.empty() || out_node.empty()) {
        result.error = "failed to assign service roles";
        return result;
    }

    if (!add_service(runtime_role::tokenizer, tok_node) ||
            !add_service(runtime_role::embedding, emb_node) ||
            !add_service(runtime_role::output_head, out_node) ||
            !add_service(runtime_role::sampler, smp_node)) {
        return result;
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
        (void) stg_desc;
        (void) weight_bytes;
        (void) n_layers;

        runtime_role_assignment stage{};
        stage.role        = runtime_role::pipeline_stage;
        stage.node_id     = la.node_id;
        stage.host        = it->second.host;
        stage.http_port   = it->second.http_port;
        stage.layer_start = la.layer_start;
        stage.layer_end   = la.layer_end;
        stage.stage_index = stage_idx++;
        stage.score       = la.score;
        result.graph.assignments.push_back(std::move(stage));
    }

    result.success = true;
    return result;
}
