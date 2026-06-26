#pragma once

#include "dist_common.h"
#include "orchestrator/layout_planner/layout_planner.h"
#include "test_layout_common.h"

#include <string>
#include <vector>

inline dist_node_info make_opt_node(
        const std::string & id,
        double score,
        double ram_gb,
        double vram_gb,
        const std::string & backend = "cuda",
        bool online = true) {
    dist_node_info n;
    n.node_id = id;
    n.online  = online;
    n.score   = score;
    n.performance.score      = score;
    n.performance.decode_tps = score;
    n.performance.prefill_tps = score * 1.5;
    n.memory.free_ram_bytes  = static_cast<uint64_t>(ram_gb * 1024.0 * 1024.0 * 1024.0);
    n.memory.free_vram_bytes = static_cast<uint64_t>(vram_gb * 1024.0 * 1024.0 * 1024.0);
    n.memory.has_gpu         = vram_gb > 0.1;
    n.caps.has_gpu           = n.memory.has_gpu;
    n.caps.gpu_backend       = n.memory.has_gpu ? backend : "cpu";
    return n;
}

inline desired_model_layout layout_for_nodes(
        const std::string & model_id,
        const model_manifest & manifest,
        const std::vector<dist_node_info> & nodes,
        int32_t n_ctx = 4096) {
    std::vector<layout_node_input> inputs;
    inputs.reserve(nodes.size());
    for (const auto & node : nodes) {
        if (node.online) {
            inputs.push_back(layout_node_from_dist(node));
        }
    }
    const auto built = build_desired_layout(model_id, manifest, inputs, n_ctx);
    return built.success ? built.layout : desired_model_layout{};
}
