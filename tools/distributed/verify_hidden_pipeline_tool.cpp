#include "verification/hidden_pipeline_parity.h"
#include "verification/decode_loop_parity.h"
#include "runtime_debug/tensor_stats.h"

#include "nlohmann/json.hpp"

#include <cstdio>
#include <string>

using json = nlohmann::json;

int main(int argc, char ** argv) {
    if (argc < 5) {
        fprintf(stderr, "usage: %s MODEL PRODUCER_TRACE.jsonl PROMPT BOUNDARY_LAYER_END [--boundary-index N]\n", argv[0]);
        return 2;
    }
    const std::string model_path = argv[1];
    const std::string entry_trace = argv[2];
    const std::string prompt      = argv[3];
    const int32_t layer_end       = std::stoi(argv[4]);
    int32_t boundary_index = -1;
    for (int i = 5; i + 1 < argc; ++i) {
        if (std::string(argv[i]) == "--boundary-index") {
            boundary_index = std::stoi(argv[i + 1]);
        }
    }

    const auto trace = parse_trace_jsonl(entry_trace);
    const auto result = verify_hidden_at_layer_boundary(model_path, prompt, layer_end, trace);

    std::string message = result.message;
    if (boundary_index >= 0) {
        if (result.ok) {
            message = result.exact_sha_match ?
                    "hidden sha256 matches at boundary #" + std::to_string(boundary_index) :
                    "hidden aggregate stats match at boundary #" + std::to_string(boundary_index);
        } else if (result.message.find("missing") != std::string::npos ||
                result.message.find("not comparable") != std::string::npos) {
            message = result.message + " at boundary #" + std::to_string(boundary_index);
        } else {
            message = "hidden stats differ at boundary #" + std::to_string(boundary_index);
        }
    }
    json out = {
        { "ok", result.ok },
        { "message", message },
        { "verification_kind", "pipeline_boundary" },
        { "boundary_index", boundary_index },
        { "boundary_layer_end", layer_end },
        { "exact_sha_match", result.exact_sha_match },
        { "aggregate_match", result.aggregate_match },
        { "reference_stats", json::parse(tensor_stats_json(result.reference_stats)) },
        { "producer_stats", json::parse(tensor_stats_json(result.producer_stats)) },
        { "metrics", json::parse(parity_metrics_json(result.metrics)) },
    };
    if (boundary_index < 0) {
        out["raw_message"] = result.message;
    }
    printf("%s\n", out.dump(2).c_str());
    return result.ok ? 0 : 1;
}
