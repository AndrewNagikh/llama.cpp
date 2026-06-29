#include "layer_trace.h"

#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <sstream>

static bool env_truthy(const char * name) {
    const char * v = std::getenv(name);
    if (v == nullptr || v[0] == '\0') {
        return false;
    }
    return std::strcmp(v, "0") != 0 &&
           std::strcmp(v, "false") != 0 &&
           std::strcmp(v, "FALSE") != 0 &&
           std::strcmp(v, "no") != 0 &&
           std::strcmp(v, "NO") != 0;
}

layer_trace_config layer_trace_load_config() {
    layer_trace_config cfg{};
    cfg.enabled         = env_truthy("LLAMA_LAYER_TRACE");
    cfg.capture_input   = env_truthy("LLAMA_LAYER_TRACE_INPUT");
    cfg.dump_raw_bins   = env_truthy("LLAMA_LAYER_TRACE_RAW");
    cfg.trace_block_ops = env_truthy("LLAMA_LAYER_TRACE_OPS");
    cfg.dump_graph      = env_truthy("LLAMA_GRAPH_DUMP");

    if (const char * block = std::getenv("LLAMA_LAYER_TRACE_BLOCK")) {
        cfg.trace_block_layer = std::atoi(block);
    }

    if (const char * dir = std::getenv("LLAMA_LAYER_TRACE_DIR")) {
        cfg.trace_dir = dir;
    } else if (const char * models = std::getenv("MODELS_DIR")) {
        cfg.trace_dir = std::string(models) + "/layer_trace";
    } else {
        cfg.trace_dir = "/tmp/layer_trace";
    }

    return cfg;
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

std::string layer_boundary_dump_json(const layer_boundary_dump & dump) {
    std::ostringstream os;
    os << "{"
       << "\"layer\":" << dump.layer_index << ","
       << "\"token_index\":" << dump.token_index << ","
       << "\"n_embd\":" << dump.n_embd << ","
       << "\"input_ok\":" << (dump.input_ok ? "true" : "false") << ","
       << "\"output_ok\":" << (dump.output_ok ? "true" : "false") << ","
       << "\"input\":" << tensor_stats_json(dump.input) << ","
       << "\"output\":" << tensor_stats_json(dump.output) << ","
       << "\"parity\":" << parity_metrics_json(dump.parity) << ","
       << "\"kv_seq_max\":" << dump.kv_seq_max << ","
       << "\"positions\":[";
    for (size_t i = 0; i < dump.positions.size(); ++i) {
        if (i > 0) {
            os << ',';
        }
        os << dump.positions[i];
    }
    os << "]}";
    return os.str();
}

void layer_trace_write_boundary(
        const layer_trace_config & cfg,
        const std::string & run_label,
        const layer_boundary_dump & dump) {
    if (!cfg.enabled) {
        return;
    }

    std::error_code ec;
    std::filesystem::create_directories(cfg.trace_dir, ec);

    const std::string path = cfg.trace_dir + "/" + run_label + "_layer_" +
            std::to_string(dump.layer_index) + ".jsonl";
    std::ofstream out(path, std::ios::app);
    if (!out) {
        return;
    }
    out << layer_boundary_dump_json(dump) << '\n';
}

void layer_trace_write_graph(
        const layer_trace_config & cfg,
        const std::string & run_label,
        const int32_t layer_start,
        const int32_t layer_end,
        const std::string & graph_json) {
    if (!cfg.enabled && !cfg.dump_graph) {
        return;
    }

    std::error_code ec;
    std::filesystem::create_directories(cfg.trace_dir, ec);

    const std::string path = cfg.trace_dir + "/" + run_label + "_graph_" +
            std::to_string(layer_start) + "_" + std::to_string(layer_end) + ".json";
    std::ofstream out(path);
    if (!out) {
        return;
    }
    out << graph_json << '\n';
}
