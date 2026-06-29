#include "graph_equivalence.h"

#include "llama-distributed.h"

#include "../../../src/llama-ext.h"

#include "ggml.h"

#include <map>
#include <sstream>

static bool is_graph_meta_op(const ggml_op op) {
    return op == GGML_OP_NONE || op == GGML_OP_VIEW || op == GGML_OP_RESHAPE ||
           op == GGML_OP_PERMUTE || op == GGML_OP_TRANSPOSE;
}

graph_summary graph_summarize_layer(
        llama_context * ctx,
        const int32_t layer_start,
        const int32_t layer_end,
        const uint32_t n_tokens) {
    graph_summary summary{};
    summary.layer_start = layer_start;
    summary.layer_end   = layer_end;
    summary.n_tokens    = (int) n_tokens;

    if (!ctx) {
        return summary;
    }

    llama_set_layer_range(ctx, layer_start, layer_end);
    ggml_cgraph * gf = llama_graph_reserve(ctx, n_tokens, 1, n_tokens);
    if (!gf) {
        return summary;
    }

    summary.n_nodes = ggml_graph_n_nodes(gf);
    std::map<std::string, int> hist;

    for (int i = 0; i < summary.n_nodes; ++i) {
        ggml_tensor * node = ggml_graph_node(gf, i);
        if (!node) {
            continue;
        }

        graph_node_info info{};
        info.index = i;
        info.op    = ggml_op_name(node->op);
        info.name  = node->name[0] != '\0' ? std::string(node->name) : std::string{};
        info.ne0   = node->ne[0];
        info.ne1   = node->ne[1];
        summary.nodes.push_back(info);

        if (!is_graph_meta_op(node->op)) {
            ++summary.n_compute_nodes;
            hist[info.op]++;
        }
    }

    for (const auto & kv : hist) {
        summary.op_histogram.push_back(kv.first + ":" + std::to_string(kv.second));
    }

    return summary;
}

graph_equivalence_result compare_layer_graphs(
        llama_context * mono_ctx,
        llama_context * worker_ctx,
        const int32_t layer_start,
        const int32_t layer_end,
        const uint32_t n_tokens) {
    graph_equivalence_result result{};
    result.mono   = graph_summarize_layer(mono_ctx, layer_start, layer_end, n_tokens);
    result.worker = graph_summarize_layer(worker_ctx, layer_start, layer_end, n_tokens);

    if (result.mono.n_nodes != result.worker.n_nodes) {
        result.match   = false;
        result.message = "graph node count differs: mono=" + std::to_string(result.mono.n_nodes) +
                         " worker=" + std::to_string(result.worker.n_nodes);
        return result;
    }

    if (result.mono.n_compute_nodes != result.worker.n_compute_nodes) {
        result.match   = false;
        result.message = "compute node count differs: mono=" +
                         std::to_string(result.mono.n_compute_nodes) +
                         " worker=" + std::to_string(result.worker.n_compute_nodes);
        return result;
    }

    for (size_t i = 0; i < result.mono.nodes.size(); ++i) {
        const auto & a = result.mono.nodes[i];
        const auto & b = result.worker.nodes[i];
        if (a.op != b.op) {
            result.match   = false;
            result.message = "op mismatch at node " + std::to_string(i) + ": mono=" + a.op +
                             " worker=" + b.op;
            return result;
        }
    }

    result.match   = true;
    result.message = "graph structure match";
    return result;
}

static void json_escape(std::ostringstream & os, const std::string & s) {
    os << '"';
    for (char c : s) {
        if (c == '"') {
            os << "\\\"";
        } else if (c == '\\') {
            os << "\\\\";
        } else {
            os << c;
        }
    }
    os << '"';
}

std::string graph_summary_json(const graph_summary & summary) {
    std::ostringstream os;
    os << "{"
       << "\"layer_start\":" << summary.layer_start << ","
       << "\"layer_end\":" << summary.layer_end << ","
       << "\"n_tokens\":" << summary.n_tokens << ","
       << "\"n_nodes\":" << summary.n_nodes << ","
       << "\"n_compute_nodes\":" << summary.n_compute_nodes << ","
       << "\"op_histogram\":[";
    for (size_t i = 0; i < summary.op_histogram.size(); ++i) {
        if (i > 0) {
            os << ',';
        }
        json_escape(os, summary.op_histogram[i]);
    }
    os << "],\"nodes\":[";
    for (size_t i = 0; i < summary.nodes.size(); ++i) {
        if (i > 0) {
            os << ',';
        }
        const auto & n = summary.nodes[i];
        os << "{"
           << "\"index\":" << n.index << ","
           << "\"op\":";
        json_escape(os, n.op);
        os << ",\"name\":";
        json_escape(os, n.name);
        os << ",\"ne0\":" << n.ne0 << ",\"ne1\":" << n.ne1 << "}";
    }
    os << "]}";
    return os.str();
}
