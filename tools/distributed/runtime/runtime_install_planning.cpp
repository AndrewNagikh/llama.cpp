#include "runtime_install_planning.h"

#include "runtime_role.h"

#include <algorithm>
#include <set>
#include <utility>

namespace {

void push_unique(std::vector<std::string> & out, const std::string & node_id) {
    if (node_id.empty()) {
        return;
    }
    if (std::find(out.begin(), out.end(), node_id) == out.end()) {
        out.push_back(node_id);
    }
}

std::string pick_or_fallback(const std::string & primary, const std::string & fallback) {
    return !primary.empty() ? primary : fallback;
}

} // namespace

runtime_install_node_map runtime_install_node_map_from_graph(const runtime_graph & graph) {
    runtime_install_node_map map{};

    std::set<std::string> seen_nodes;
    for (const runtime_role_assignment & a : graph.assignments) {
        if (a.node_id.empty()) {
            continue;
        }
        if (seen_nodes.insert(a.node_id).second) {
            map.all_nodes.push_back(a.node_id);
        }
        switch (a.role) {
            case runtime_role::tokenizer:
                map.tokenizer_node = a.node_id;
                break;
            case runtime_role::embedding:
                map.embedding_node = a.node_id;
                break;
            case runtime_role::output_head:
                map.output_head_node = a.node_id;
                break;
            case runtime_role::pipeline_stage:
                if (map.first_pipeline_node.empty()) {
                    map.first_pipeline_node = a.node_id;
                }
                map.last_pipeline_node = a.node_id;
                break;
            default:
                break;
        }
    }

    return map;
}

runtime_install_node_map runtime_install_node_map_legacy(
        const std::string & entry_node,
        const std::string & final_node,
        const std::vector<std::string> & pipeline_nodes) {
    runtime_install_node_map map{};
    map.tokenizer_node       = entry_node;
    map.embedding_node       = entry_node;
    map.output_head_node     = final_node;
    map.first_pipeline_node  = entry_node;
    map.last_pipeline_node   = final_node;
    map.all_nodes            = pipeline_nodes;
    push_unique(map.all_nodes, entry_node);
    push_unique(map.all_nodes, final_node);
    return map;
}

std::vector<std::string> nodes_for_runtime_blob_deploy(
        const blob_deploy_target deploy,
        const tensor_semantic_role role,
        const runtime_install_node_map & map) {
    std::vector<std::string> nodes;
    switch (deploy) {
        case blob_deploy_target::all_nodes:
            nodes = map.all_nodes;
            break;
        case blob_deploy_target::entry_node: {
            std::string target;
            switch (role) {
                case tensor_semantic_role::metadata:
                    target = pick_or_fallback(map.tokenizer_node, map.first_pipeline_node);
                    break;
                case tensor_semantic_role::embedding:
                case tensor_semantic_role::input_norm:
                case tensor_semantic_role::rotary:
                    target = pick_or_fallback(map.embedding_node, map.first_pipeline_node);
                    break;
                default:
                    target = pick_or_fallback(map.embedding_node, map.first_pipeline_node);
                    if (target.empty()) {
                        target = map.first_pipeline_node;
                    }
                    break;
            }
            push_unique(nodes, target);
            break;
        }
        case blob_deploy_target::final_node: {
            std::string target;
            switch (role) {
                case tensor_semantic_role::output_head:
                case tensor_semantic_role::output_norm:
                    target = pick_or_fallback(map.output_head_node, map.last_pipeline_node);
                    break;
                default:
                    target = pick_or_fallback(map.output_head_node, map.last_pipeline_node);
                    break;
            }
            push_unique(nodes, target);
            break;
        }
        case blob_deploy_target::none:
            break;
    }
    return nodes;
}

static void add_resource_assignment(
        runtime_resource_install_plan & plan,
        const std::string & resource_id,
        const runtime_role service_role,
        const std::string & node_id,
        const bool required) {
    if (resource_id.empty() || node_id.empty()) {
        return;
    }
    for (const runtime_resource_install_assignment & existing : plan.assignments) {
        if (existing.resource_id == resource_id &&
                existing.service_role == service_role &&
                existing.node_id == node_id) {
            return;
        }
    }
    runtime_resource_install_assignment a{};
    a.resource_id = resource_id;
    a.service_role = service_role;
    a.node_id = node_id;
    a.required = required;
    plan.assignments.push_back(std::move(a));
}

static std::vector<std::string> graph_nodes_for_service(
        const runtime_graph & graph,
        const runtime_role role) {
    std::vector<std::string> nodes;
    for (const runtime_role_assignment & assignment : graph.assignments) {
        if (assignment.role == role) {
            push_unique(nodes, assignment.node_id);
        }
    }
    return nodes;
}

runtime_resource_install_plan build_runtime_resource_install_plan(
        const runtime_descriptor & desc,
        const runtime_graph & graph) {
    runtime_resource_install_plan plan{};

    const runtime_execution_graph_validation graph_validation =
            validate_runtime_graph_against_descriptor(desc, graph);
    if (!graph_validation.ok()) {
        plan.error = graph_validation.errors.empty() ?
                "invalid runtime graph" :
                graph_validation.errors.front();
        return plan;
    }

    for (const runtime_service_descriptor & service : desc.services) {
        const std::vector<std::string> service_nodes =
                graph_nodes_for_service(graph, service.role);
        if (service_nodes.empty()) {
            plan.error = "missing service placement: " + service.name;
            return plan;
        }

        for (const std::string & node_id : service_nodes) {
            for (const std::string & resource_id : service.required_resources) {
                add_resource_assignment(
                        plan, resource_id, service.role, node_id, true);
            }
            for (const std::string & resource_id : service.optional_resources) {
                add_resource_assignment(
                        plan, resource_id, service.role, node_id, false);
            }
        }
    }

    plan.success = true;
    return plan;
}

bool runtime_resource_plan_has(
        const runtime_resource_install_plan & plan,
        const std::string & resource_id,
        const runtime_role service_role,
        const std::string & node_id) {
    for (const runtime_resource_install_assignment & assignment : plan.assignments) {
        if (assignment.resource_id == resource_id &&
                assignment.service_role == service_role &&
                assignment.node_id == node_id) {
            return true;
        }
    }
    return false;
}

runtime_resource_install_plan validate_runtime_resource_install_plan(
        const runtime_descriptor & desc,
        const runtime_graph & graph,
        const runtime_resource_install_plan & plan) {
    runtime_resource_install_plan out{};
    if (!plan.success) {
        out.error = plan.error.empty() ? "install plan is not successful" : plan.error;
        return out;
    }

    const runtime_resource_install_plan expected =
            build_runtime_resource_install_plan(desc, graph);
    if (!expected.success) {
        out.error = expected.error;
        return out;
    }

    for (const runtime_resource_install_assignment & assignment : plan.assignments) {
        if (find_runtime_resource(desc, assignment.resource_id) == nullptr) {
            out.error = "unknown resource in install plan: " + assignment.resource_id;
            return out;
        }
        if (find_runtime_service(desc, assignment.service_role) == nullptr) {
            out.error = "unknown service in install plan: " +
                    runtime_role_name(assignment.service_role);
            return out;
        }
        const std::vector<std::string> service_nodes =
                graph_nodes_for_service(graph, assignment.service_role);
        if (std::find(service_nodes.begin(), service_nodes.end(), assignment.node_id) ==
                service_nodes.end()) {
            out.error = "resource assigned to wrong service node: " +
                    assignment.resource_id;
            return out;
        }
    }

    for (const runtime_resource_install_assignment & assignment : expected.assignments) {
        if (!assignment.required) {
            continue;
        }
        if (!runtime_resource_plan_has(
                    plan,
                    assignment.resource_id,
                    assignment.service_role,
                    assignment.node_id)) {
            out.error = "missing required resource assignment: " +
                    assignment.resource_id;
            return out;
        }
    }

    out.success = true;
    out.assignments = plan.assignments;
    return out;
}
