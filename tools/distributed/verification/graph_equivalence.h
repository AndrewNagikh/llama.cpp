#pragma once

#include "llama.h"

#include <string>
#include <vector>

struct graph_node_info {
    int index = -1;
    std::string op;
    std::string name;
    int64_t ne0 = 0;
    int64_t ne1 = 0;
};

struct graph_summary {
    int32_t layer_start = 0;
    int32_t layer_end   = 0;
    int n_tokens        = 0;
    int n_nodes         = 0;
    int n_compute_nodes = 0;
    std::vector<graph_node_info> nodes;
    std::vector<std::string> op_histogram;
};

graph_summary graph_summarize_layer(
        llama_context * ctx,
        int32_t layer_start,
        int32_t layer_end,
        uint32_t n_tokens);

struct graph_equivalence_result {
    bool match = false;
    graph_summary mono{};
    graph_summary worker{};
    std::string message;
};

graph_equivalence_result compare_layer_graphs(
        llama_context * mono_ctx,
        llama_context * worker_ctx,
        int32_t layer_start,
        int32_t layer_end,
        uint32_t n_tokens);

std::string graph_summary_json(const graph_summary & summary);
