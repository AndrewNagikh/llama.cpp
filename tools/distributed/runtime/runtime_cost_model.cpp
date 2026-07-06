#include "runtime_cost_model.h"

#include <algorithm>
#include <cmath>

runtime_planner_node runtime_planner_node_from_dist(const dist_node_info & node) {
    runtime_planner_node n{};
    n.node_id          = node.node_id;
    n.score            = node.score > 0 ? node.score : 1.0;
    n.backend          = node.caps.gpu_backend.empty() ? "cpu" : node.caps.gpu_backend;
    n.has_gpu          = node.memory.has_gpu;
    n.cpu_budget_bytes = node.memory.free_ram_bytes;
    n.gpu_budget_bytes = node.memory.free_vram_bytes;
    n.cpu_score        = std::max(1.0, static_cast<double>(node.cpu.logical_cores > 0 ? node.cpu.logical_cores : node.caps.cpu_threads));
    const double ram_gb = dist_bytes_to_gb(node.memory.free_ram_bytes);
    n.memory_bw_score  = std::max(1.0, ram_gb);
    return n;
}

uint64_t runtime_node_budget_bytes(const runtime_planner_node & node, const bool prefers_gpu) {
    if (prefers_gpu && node.has_gpu && node.gpu_budget_bytes > 0) {
        return node.gpu_budget_bytes;
    }
    return node.cpu_budget_bytes;
}

double runtime_pipeline_service_penalty(
        const runtime_planner_node & node,
        const runtime_role role) {
    double penalty = 1.0 + 0.15 * static_cast<double>(std::max(node.pipeline_layers, 0));
    if (role == runtime_role::tokenizer || role == runtime_role::embedding) {
        if (node.is_first_pipeline_stage) {
            penalty *= 4.0;
        }
    }
    if (role == runtime_role::output_head && node.is_last_pipeline_stage) {
        penalty *= 3.0;
    }
    return penalty;
}

static double cost_from_budget(const runtime_planner_node & node, const runtime_role_descriptor & desc) {
    const uint64_t budget = runtime_node_budget_bytes(node, desc.prefers_gpu);
    if (budget < desc.required_memory_bytes) {
        return 1e12;
    }
    return static_cast<double>(desc.required_memory_bytes) / static_cast<double>(std::max<uint64_t>(budget, 1));
}

double runtime_cost_tokenizer(const runtime_planner_node & node, const runtime_role_descriptor & desc) {
    const double mem_penalty = cost_from_budget(node, desc);
    return mem_penalty * runtime_pipeline_service_penalty(node, runtime_role::tokenizer) / node.cpu_score;
}

double runtime_cost_embedding(const runtime_planner_node & node, const runtime_role_descriptor & desc) {
    const double mem_penalty = cost_from_budget(node, desc);
    const double gpu_bonus   = node.has_gpu ? (1.0 / std::max(node.score, 1.0)) : 1.0;
    return mem_penalty * gpu_bonus * runtime_pipeline_service_penalty(node, runtime_role::embedding) /
            node.memory_bw_score;
}

double runtime_cost_pipeline_stage(
        const runtime_planner_node & node,
        const runtime_role_descriptor & desc,
        const int32_t layer_count) {
    runtime_role_descriptor scaled = desc;
    scaled.required_memory_bytes = std::max<uint64_t>(
            desc.required_memory_bytes / 8,
            static_cast<uint64_t>(layer_count) * (desc.required_memory_bytes / std::max<int32_t>(layer_count, 1)));
    const double mem_penalty = cost_from_budget(node, scaled);
    const double gpu_bonus   = node.has_gpu ? (1.0 / std::max(node.score, 1.0)) : 2.0;
    return mem_penalty * gpu_bonus;
}

double runtime_cost_output_head(const runtime_planner_node & node, const runtime_role_descriptor & desc) {
    const double mem_penalty = cost_from_budget(node, desc);
    const double gpu_bonus   = node.has_gpu ? (1.0 / std::max(node.score, 1.0)) : 3.0;
    return mem_penalty * gpu_bonus * runtime_pipeline_service_penalty(node, runtime_role::output_head);
}

double runtime_cost_sampler(const runtime_planner_node & node, const runtime_role_descriptor & desc) {
    const double mem_penalty = cost_from_budget(node, desc);
    return mem_penalty / node.cpu_score;
}

double runtime_cost_for_role(
        const runtime_role role,
        const runtime_planner_node & node,
        const runtime_role_descriptor & desc,
        const int32_t layer_count) {
    switch (role) {
        case runtime_role::tokenizer:      return runtime_cost_tokenizer(node, desc);
        case runtime_role::embedding:      return runtime_cost_embedding(node, desc);
        case runtime_role::pipeline_stage: return runtime_cost_pipeline_stage(node, desc, layer_count);
        case runtime_role::output_head:  return runtime_cost_output_head(node, desc);
        case runtime_role::sampler:        return runtime_cost_sampler(node, desc);
        default:                           return 1e12;
    }
}
