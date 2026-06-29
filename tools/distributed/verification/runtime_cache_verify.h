#pragma once

#include "llama.h"

#include <cstdint>
#include <string>

struct graph_reserve_snapshot {
    uint32_t n_tokens     = 0;
    int32_t layer_start   = 0;
    int32_t layer_end     = 0;
    int n_nodes           = 0;
    int n_compute_nodes   = 0;
    uintptr_t graph_ptr   = 0;
    uintptr_t first_node_ptr = 0;
};

struct runtime_cache_diag {
    graph_reserve_snapshot first{};
    graph_reserve_snapshot second{};
    bool same_node_count  = false;
    bool same_graph_ptr   = false;
    bool same_first_node  = false;
    std::string message;
};

runtime_cache_diag diagnose_runtime_graph_cache(
        llama_context * ctx,
        int32_t layer_start,
        int32_t layer_end,
        uint32_t n_tokens);

std::string runtime_cache_diag_json(const runtime_cache_diag & diag);
