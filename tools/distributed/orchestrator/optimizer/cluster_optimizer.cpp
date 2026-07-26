#include "cluster_optimizer.h"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <vector>

using json = nlohmann::json;

std::string node_role_to_string(const node_role role) {
    switch (role) {
        case node_role::active:       return "ACTIVE";
        case node_role::storage_only: return "STORAGE_ONLY";
        case node_role::standby:      return "STANDBY";
    }
    return "ACTIVE";
}

node_role node_role_from_string(const std::string & s) {
    if (s == "STORAGE_ONLY") return node_role::storage_only;
    if (s == "STANDBY")      return node_role::standby;
    return node_role::active;
}

json optimizer_node::to_json() const {
    return {
        { "node_id", node_id },
        { "score", score },
        { "decode_tps", decode_tps },
        { "prefill_tps", prefill_tps },
        { "free_ram", free_ram },
        { "free_vram", free_vram },
        { "role", node_role_to_string(role) },
        { "online", online },
    };
}

std::string optimizer_decision_to_string(const optimizer_decision decision) {
    switch (decision) {
        case optimizer_decision::keep:               return "KEEP";
        case optimizer_decision::rebalance:          return "REBALANCE";
        case optimizer_decision::storage_only_assign: return "STORAGE_ONLY";
    }
    return "KEEP";
}

json optimization_result::to_json() const {
    json roles = json::object();
    for (const auto & kv : node_roles) {
        roles[kv.first] = node_role_to_string(kv.second);
    }

    json nodes_json = json::object();
    for (const auto & kv : nodes) {
        nodes_json[kv.first] = kv.second.to_json();
    }

    return {
        { "model_id", model_id },
        { "decision", optimizer_decision_to_string(decision) },
        { "better", better },
        { "reason", reason },
        { "current_decode_tps", current_estimate.estimated_decode_tps },
        { "candidate_decode_tps", candidate_estimate.estimated_decode_tps },
        { "current_prefill_tps", current_estimate.estimated_prefill_tps },
        { "candidate_prefill_tps", candidate_estimate.estimated_prefill_tps },
        { "rebalance_cost_bytes", candidate_estimate.rebalance_cost_bytes },
        { "current_estimate", current_estimate.to_json() },
        { "candidate_estimate", candidate_estimate.to_json() },
        { "current_layout", current_layout.to_json() },
        { "candidate_layout", candidate_layout.to_json() },
        { "node_roles", roles },
        { "nodes", nodes_json },
    };
}

optimizer_node optimizer_node_from_dist(const dist_node_info & node) {
    optimizer_node on;
    on.node_id    = node.node_id;
    on.score      = node.score > 0.0 ? node.score : node.performance.score;
    on.decode_tps = node.performance.decode_tps > 0.0
            ? node.performance.decode_tps
            : on.score;
    on.prefill_tps = node.performance.prefill_tps > 0.0
            ? node.performance.prefill_tps
            : on.decode_tps;
    on.free_ram   = node.memory.free_ram_bytes;
    on.free_vram  = node.memory.free_vram_bytes;
    on.online     = node.online;
    on.role       = node_role::active;
    return on;
}

std::map<std::string, node_role> classify_node_roles(
        const std::vector<optimizer_node> & nodes,
        const optimizer_policy & policy,
        const double cluster_median_score) {
    std::map<std::string, node_role> roles;
    const double threshold = std::max(cluster_median_score * policy.storage_only_score_ratio, 1.0);

    for (const auto & node : nodes) {
        if (!policy.allow_storage_only_nodes) {
            roles[node.node_id] = node_role::active;
            continue;
        }
        if (!node.online) {
            roles[node.node_id] = node_role::standby;
            continue;
        }
        const bool enough_memory = node.free_ram > 0 || node.free_vram > 0;
        if (enough_memory && node.score > 0.0 && node.score < threshold) {
            roles[node.node_id] = node_role::storage_only;
        } else {
            roles[node.node_id] = node_role::active;
        }
    }
    return roles;
}

static double median_score(const std::vector<optimizer_node> & nodes) {
    std::vector<double> scores;
    scores.reserve(nodes.size());
    for (const auto & n : nodes) {
        if (n.online && n.score > 0.0) {
            scores.push_back(n.score);
        }
    }
    if (scores.empty()) {
        return 1.0;
    }
    std::sort(scores.begin(), scores.end());
    return scores[scores.size() / 2];
}

static std::map<std::string, double> decode_map(const std::vector<optimizer_node> & nodes) {
    std::map<std::string, double> out;
    for (const auto & n : nodes) {
        out[n.node_id] = n.decode_tps;
    }
    return out;
}

static std::map<std::string, double> prefill_map(const std::vector<optimizer_node> & nodes) {
    std::map<std::string, double> out;
    for (const auto & n : nodes) {
        out[n.node_id] = n.prefill_tps;
    }
    return out;
}

optimization_result run_cluster_optimization(
        const std::string & model_id,
        const model_manifest & manifest,
        const desired_model_layout * current_layout,
        const std::vector<dist_node_info> & cluster_nodes,
        const optimizer_policy & policy,
        const int32_t n_ctx) {
    optimization_result result;
    result.model_id = model_id;

    std::vector<optimizer_node> opt_nodes;
    opt_nodes.reserve(cluster_nodes.size());
    for (const auto & node : cluster_nodes) {
        if (!node.online) {
            continue;
        }
        opt_nodes.push_back(optimizer_node_from_dist(node));
    }

    if (opt_nodes.empty()) {
        result.reason = "no online nodes";
        return result;
    }

    const double median = median_score(opt_nodes);
    result.node_roles = classify_node_roles(opt_nodes, policy, median);

    for (auto & n : opt_nodes) {
        const auto it = result.node_roles.find(n.node_id);
        if (it != result.node_roles.end()) {
            n.role = it->second;
        }
        result.nodes[n.node_id] = n;
    }

    std::vector<layout_node_input> active_inputs;
    active_inputs.reserve(opt_nodes.size());
    for (const auto & node : cluster_nodes) {
        if (!node.online) {
            continue;
        }
        const auto role_it = result.node_roles.find(node.node_id);
        if (role_it != result.node_roles.end() &&
                role_it->second == node_role::storage_only) {
            continue;
        }
        active_inputs.push_back(layout_node_from_dist(node));
    }

    if (current_layout != nullptr) {
        result.current_layout = *current_layout;
        result.current_estimate = estimate_layout_performance(
                *current_layout, decode_map(opt_nodes), prefill_map(opt_nodes));
    }

    if (active_inputs.empty()) {
        result.reason = "no ACTIVE nodes for inference";
        return result;
    }

    // Role hysteresis against the layout currently in place: the optimizer
    // already gates on a throughput improvement threshold, but that compares
    // estimates -- this keeps the node *ordering* itself from flip-flopping on
    // score noise before the estimate is even computed.
    const layout_build_result built = build_desired_layout(
            model_id, manifest, active_inputs, n_ctx, current_layout);
    if (!built.success || !built.layout.fits_cluster) {
        result.reason = built.error.empty() ? "candidate layout does not fit" : built.error;
        if (current_layout != nullptr) {
            result.candidate_layout = *current_layout;
            result.candidate_estimate = result.current_estimate;
        }
        return result;
    }

    result.candidate_layout = built.layout;
    result.candidate_estimate = estimate_layout_performance(
            built.layout, decode_map(opt_nodes), prefill_map(opt_nodes));

    if (current_layout != nullptr) {
        result.candidate_estimate.rebalance_cost_bytes = estimate_rebalance_cost_bytes(
                *current_layout, built.layout, manifest);
    }

    if (current_layout == nullptr) {
        result.decision = optimizer_decision::rebalance;
        result.better = true;
        result.reason = "initial layout required";
        result.candidate_estimate.better_than_current = true;
        return result;
    }

    result.better = layout_estimate_is_better(
            result.current_estimate,
            result.candidate_estimate,
            policy.min_decode_improvement_percent,
            policy.min_prefill_improvement_percent,
            policy.max_rebalance_cost_bytes);

    result.candidate_estimate.better_than_current = result.better;

    if (result.better) {
        result.decision = optimizer_decision::rebalance;
        result.reason = "candidate layout improves throughput above policy threshold";
    } else {
        result.decision = optimizer_decision::keep;
        result.candidate_layout = *current_layout;
        result.candidate_estimate = result.current_estimate;
        result.reason = "current layout is sufficient; rebalance not justified";

        const bool has_storage_only = std::any_of(
                result.node_roles.begin(), result.node_roles.end(),
                [](const auto & kv) { return kv.second == node_role::storage_only; });
        if (has_storage_only) {
            result.decision = optimizer_decision::storage_only_assign;
            result.reason = "weak node assigned STORAGE_ONLY; inference layout unchanged";
        }
    }

    return result;
}
