#include "runtime_cache_verify.h"

#include "graph_equivalence.h"

#include "../../../src/llama-ext.h"

#include "ggml.h"
#include "llama-distributed.h"

#include <sstream>

static graph_reserve_snapshot reserve_snapshot(
        llama_context * ctx,
        const int32_t layer_start,
        const int32_t layer_end,
        const uint32_t n_tokens) {
    graph_reserve_snapshot snap{};
    snap.n_tokens     = n_tokens;
    snap.layer_start  = layer_start;
    snap.layer_end    = layer_end;

    if (!ctx) {
        return snap;
    }

    llama_set_layer_range(ctx, layer_start, layer_end);
    ggml_cgraph * gf = llama_graph_reserve(ctx, n_tokens, 1, n_tokens);
    if (!gf) {
        return snap;
    }

    snap.graph_ptr = reinterpret_cast<uintptr_t>(gf);
    snap.n_nodes   = ggml_graph_n_nodes(gf);

    for (int i = 0; i < snap.n_nodes; ++i) {
        ggml_tensor * node = ggml_graph_node(gf, i);
        if (!node) {
            continue;
        }
        if (node->op != GGML_OP_NONE && node->op != GGML_OP_VIEW && node->op != GGML_OP_RESHAPE &&
            node->op != GGML_OP_PERMUTE && node->op != GGML_OP_TRANSPOSE) {
            ++snap.n_compute_nodes;
            if (snap.first_node_ptr == 0) {
                snap.first_node_ptr = reinterpret_cast<uintptr_t>(node);
            }
        }
    }

    return snap;
}

runtime_cache_diag diagnose_runtime_graph_cache(
        llama_context * ctx,
        const int32_t layer_start,
        const int32_t layer_end,
        const uint32_t n_tokens) {
    runtime_cache_diag diag{};
    diag.first  = reserve_snapshot(ctx, layer_start, layer_end, n_tokens);
    diag.second = reserve_snapshot(ctx, layer_start, layer_end, n_tokens);

    diag.same_node_count = diag.first.n_nodes == diag.second.n_nodes &&
                           diag.first.n_compute_nodes == diag.second.n_compute_nodes;
    diag.same_graph_ptr  = diag.first.graph_ptr != 0 &&
                           diag.first.graph_ptr == diag.second.graph_ptr;
    diag.same_first_node = diag.first.first_node_ptr != 0 &&
                           diag.first.first_node_ptr == diag.second.first_node_ptr;

    if (diag.same_node_count && diag.same_graph_ptr) {
        diag.message = "graph cache stable (same reservation ptr)";
    } else if (diag.same_node_count) {
        diag.message = "graph structure stable (reservation rebuilt)";
    } else {
        diag.message = "graph reservation differs between consecutive calls";
    }
    return diag;
}

static std::string snapshot_json(const graph_reserve_snapshot & s) {
    std::ostringstream os;
    os << "{"
       << "\"n_tokens\":" << s.n_tokens
       << ",\"layer_start\":" << s.layer_start
       << ",\"layer_end\":" << s.layer_end
       << ",\"n_nodes\":" << s.n_nodes
       << ",\"n_compute_nodes\":" << s.n_compute_nodes
       << ",\"graph_ptr\":" << s.graph_ptr
       << ",\"first_node_ptr\":" << s.first_node_ptr
       << "}";
    return os.str();
}

std::string runtime_cache_diag_json(const runtime_cache_diag & diag) {
    std::ostringstream os;
    os << "{"
       << "\"same_node_count\":" << (diag.same_node_count ? "true" : "false")
       << ",\"same_graph_ptr\":" << (diag.same_graph_ptr ? "true" : "false")
       << ",\"same_first_node\":" << (diag.same_first_node ? "true" : "false")
       << ",\"message\":\"" << diag.message << "\""
       << ",\"first\":" << snapshot_json(diag.first)
       << ",\"second\":" << snapshot_json(diag.second)
       << "}";
    return os.str();
}
