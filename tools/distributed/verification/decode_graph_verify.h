#pragma once

#include "graph_equivalence.h"

#include "llama.h"

#include <string>

struct decode_graph_verify_result {
    graph_equivalence_result comparison{};
    graph_summary mono{};
    graph_summary worker{};
    std::string mono_path;
    std::string worker_path;
    int32_t layer_start = 0;
    int32_t layer_end   = 0;
    uint32_t n_tokens   = 1;
    std::string graph_kind; // "prefill" | "decode"
};

decode_graph_verify_result verify_decode_graph(
        llama_context * mono_ctx,
        llama_context * worker_ctx,
        int32_t layer_start,
        int32_t layer_end,
        uint32_t n_tokens,
        const char * graph_kind = "decode");

bool write_graph_summary_json(const std::string & path, const graph_summary & summary);

std::string decode_graph_diff_json(const decode_graph_verify_result & result);
