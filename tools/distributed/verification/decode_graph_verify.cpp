#include "decode_graph_verify.h"

#include <fstream>
#include <sstream>

decode_graph_verify_result verify_decode_graph(
        llama_context * mono_ctx,
        llama_context * worker_ctx,
        const int32_t layer_start,
        const int32_t layer_end,
        const uint32_t n_tokens,
        const char * graph_kind) {
    decode_graph_verify_result result{};
    result.layer_start = layer_start;
    result.layer_end   = layer_end;
    result.n_tokens    = n_tokens;
    result.graph_kind  = graph_kind ? graph_kind : "decode";

    result.mono   = graph_summarize_layer(mono_ctx, layer_start, layer_end, n_tokens);
    result.worker = graph_summarize_layer(worker_ctx, layer_start, layer_end, n_tokens);
    result.comparison = compare_layer_graphs(
            mono_ctx, worker_ctx, layer_start, layer_end, n_tokens);
    return result;
}

bool write_graph_summary_json(const std::string & path, const graph_summary & summary) {
    std::ofstream out(path);
    if (!out) {
        return false;
    }
    out << graph_summary_json(summary);
    return static_cast<bool>(out);
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

std::string decode_graph_diff_json(const decode_graph_verify_result & result) {
    std::ostringstream os;
    os << "{"
       << "\"graph_kind\":";
    json_escape(os, result.graph_kind);
    os << ",\"layer_start\":" << result.layer_start
       << ",\"layer_end\":" << result.layer_end
       << ",\"n_tokens\":" << result.n_tokens
       << ",\"match\":" << (result.comparison.match ? "true" : "false")
       << ",\"message\":";
    json_escape(os, result.comparison.message);
    os << ",\"mono\":" << graph_summary_json(result.mono)
       << ",\"worker\":" << graph_summary_json(result.worker)
       << "}";
    return os.str();
}
