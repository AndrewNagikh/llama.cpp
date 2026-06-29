#pragma once

#include "layer_runner.h"
#include "runtime_debug/layer_trace.h"

#include <string>
#include <vector>

enum class layer_fail_kind {
    none,
    output,
    input,
};

struct layer_parity_row {
    int32_t layer_index = -1;
    bool pass           = false;
    parity_metrics output_parity{};
    parity_metrics input_parity{};
    std::string message;
};

struct layer_equivalence_diagnostics {
    int32_t fail_layer = -1;
    layer_fail_kind fail_kind = layer_fail_kind::none;

    bool embedding_input_match = false;
    bool positions_match       = false;
    bool kv_seq_match          = false;

    parity_metrics embedding_parity{};
    std::string hint;
};

struct layer_equivalence_report {
    bool all_pass              = false;
    int32_t layers_checked     = 0;
    int32_t first_fail_layer   = -1;
    std::vector<layer_parity_row> rows;
    layer_equivalence_diagnostics diagnostics;
    std::string message;
};

layer_equivalence_report verify_layer_equivalence(
        const std::string & mono_path,
        const std::string & worker_path,
        const std::string & prompt,
        int32_t max_layer = -1,
        const layer_trace_config * trace_cfg = nullptr);

std::string layer_equivalence_report_json(const layer_equivalence_report & report);
