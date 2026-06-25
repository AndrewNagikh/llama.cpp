#pragma once

#include "orchestrator/layout_planner/layout_planner.h"

#include <cstdint>
#include <string>
#include <vector>

inline model_manifest make_test_manifest(
        int32_t n_layer,
        uint64_t bytes_per_layer,
        uint32_t n_ctx = 4096) {
    model_manifest m;
    m.architecture = "llama";
    m.n_layer = static_cast<uint32_t>(n_layer);
    m.n_ctx = n_ctx;
    m.n_embd = 2048;
    m.n_vocab = 32000;

    for (int32_t i = 0; i < n_layer; ++i) {
        layer_descriptor ld;
        ld.layer_index = i;
        ld.size_bytes = bytes_per_layer;
        ld.tensors.push_back("blk." + std::to_string(i) + ".attn_norm.weight");
        m.layers.push_back(ld);
    }
    return m;
}

inline layout_node_input make_layout_node(
        const std::string & id,
        double score,
        double ram_gb,
        double vram_gb,
        const std::string & backend = "cpu") {
    layout_node_input n;
    n.node_id = id;
    n.score = score;
    n.cpu_budget_bytes = static_cast<uint64_t>(ram_gb * 1024.0 * 1024.0 * 1024.0);
    n.gpu_budget_bytes = static_cast<uint64_t>(vram_gb * 1024.0 * 1024.0 * 1024.0);
    n.has_gpu = vram_gb > 0.0;
    n.backend = n.has_gpu ? backend : "cpu";
    return n;
}
