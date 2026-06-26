#include "layout_estimator.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <map>

using json = nlohmann::json;

json layout_estimate::to_json() const {
    return {
        { "estimated_decode_tps", estimated_decode_tps },
        { "estimated_prefill_tps", estimated_prefill_tps },
        { "estimated_pipeline_latency_ms", estimated_pipeline_latency_ms },
        { "network_cost", network_cost },
        { "rebalance_cost_bytes", rebalance_cost_bytes },
        { "better_than_current", better_than_current },
    };
}

layout_estimate estimate_layout_performance(
        const desired_model_layout & layout,
        const std::map<std::string, double> & node_decode_tps,
        const std::map<std::string, double> & node_prefill_tps) {
    layout_estimate est;

    std::map<std::string, int> layer_counts;
    for (const auto & p : layout.placements) {
        if (p.node_id.empty()) {
            continue;
        }
        layer_counts[p.node_id]++;
    }
    if (layer_counts.empty()) {
        return est;
    }

    double min_decode = std::numeric_limits<double>::infinity();
    double prefill_sum = 0.0;
    int    prefill_nodes = 0;

    for (const auto & kv : layer_counts) {
        const auto decode_it = node_decode_tps.find(kv.first);
        const double decode = decode_it != node_decode_tps.end() ? decode_it->second : 1.0;
        if (decode > 0.0) {
            min_decode = std::min(min_decode, decode);
        }

        const auto prefill_it = node_prefill_tps.find(kv.first);
        const double prefill = prefill_it != node_prefill_tps.end() ? prefill_it->second : decode;
        if (prefill > 0.0) {
            prefill_sum += prefill * static_cast<double>(kv.second);
            ++prefill_nodes;
        }
    }

    if (!std::isfinite(min_decode) || min_decode <= 0.0) {
        min_decode = 1.0;
    }

    est.estimated_decode_tps = min_decode;
    est.estimated_prefill_tps = prefill_nodes > 0
            ? prefill_sum / static_cast<double>(prefill_nodes)
            : min_decode;
    est.estimated_pipeline_latency_ms = 1000.0 / std::max(min_decode, 1.0);
    est.network_cost = static_cast<double>(layer_counts.size()) * 0.5;
    return est;
}

uint64_t estimate_rebalance_cost_bytes(
        const desired_model_layout & current_layout,
        const desired_model_layout & candidate_layout,
        const model_manifest & manifest) {
    std::map<int32_t, std::string> current_map;
    for (const auto & p : current_layout.placements) {
        current_map[p.layer_index] = p.node_id;
    }

    uint64_t cost = 0;
    for (const auto & p : candidate_layout.placements) {
        const auto it = current_map.find(p.layer_index);
        if (it != current_map.end() && it->second == p.node_id) {
            continue;
        }
        const layer_byte_range range = manifest_layer_byte_range(manifest, p.layer_index);
        const uint64_t bytes = range.length > 0 ? range.length : range.size_bytes;
        cost += bytes;
        if (it != current_map.end() && !it->second.empty()) {
            cost += bytes;
        }
    }
    return cost;
}

bool layout_estimate_is_better(
        const layout_estimate & current,
        const layout_estimate & candidate,
        const double min_decode_improvement_percent,
        const double min_prefill_improvement_percent,
        const uint64_t max_rebalance_cost_bytes) {
    if (candidate.rebalance_cost_bytes > max_rebalance_cost_bytes) {
        return false;
    }

    const double decode_base = std::max(current.estimated_decode_tps, 1.0);
    const double prefill_base = std::max(current.estimated_prefill_tps, 1.0);

    const double decode_gain =
            (candidate.estimated_decode_tps - current.estimated_decode_tps) * 100.0 / decode_base;
    const double prefill_gain =
            (candidate.estimated_prefill_tps - current.estimated_prefill_tps) * 100.0 / prefill_base;

    return decode_gain >= min_decode_improvement_percent ||
           prefill_gain >= min_prefill_improvement_percent;
}
