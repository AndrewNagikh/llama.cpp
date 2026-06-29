#pragma once

#include "tensor_stats.h"

#include <cstdint>
#include <string>
#include <vector>

// Task 9.8.4 — per-layer activation tracing (observation only).

struct layer_trace_config {
    bool enabled            = false;
    bool capture_input      = false;
    bool dump_raw_bins      = false;
    bool trace_block_ops    = false;
    bool dump_graph         = false;
    int32_t trace_block_layer = -1;
    std::string trace_dir;
};

layer_trace_config layer_trace_load_config();

struct layer_boundary_dump {
    int32_t layer_index = -1;
    int32_t token_index = -1;
    int32_t n_embd      = 0;
    tensor_stats input{};
    tensor_stats output{};
    parity_metrics parity{};
    std::vector<int32_t> positions;
    int32_t kv_seq_max = -1;
    bool input_ok      = false;
    bool output_ok     = false;
};

void layer_trace_write_boundary(
        const layer_trace_config & cfg,
        const std::string & run_label,
        const layer_boundary_dump & dump);

void layer_trace_write_graph(
        const layer_trace_config & cfg,
        const std::string & run_label,
        int32_t layer_start,
        int32_t layer_end,
        const std::string & graph_json);

std::string layer_boundary_dump_json(const layer_boundary_dump & dump);
