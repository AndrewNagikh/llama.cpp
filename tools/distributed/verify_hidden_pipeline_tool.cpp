#include "verification/hidden_pipeline_parity.h"
#include "verification/decode_loop_parity.h"
#include "runtime_debug/tensor_stats.h"

#include "nlohmann/json.hpp"

#include <cstdio>
#include <string>

using json = nlohmann::json;

int main(int argc, char ** argv) {
    if (argc < 5) {
        fprintf(stderr, "usage: %s MODEL ENTRY_TRACE.jsonl PROMPT LAYER_END\n", argv[0]);
        return 2;
    }
    const std::string model_path = argv[1];
    const std::string entry_trace = argv[2];
    const std::string prompt      = argv[3];
    const int32_t layer_end       = std::stoi(argv[4]);

    const auto trace = parse_trace_jsonl(entry_trace);
    const auto result = verify_hidden_at_layer_boundary(model_path, prompt, layer_end, trace);

    json out = {
        { "ok", result.ok },
        { "message", result.message },
        { "metrics", json::parse(parity_metrics_json(result.metrics)) },
    };
    printf("%s\n", out.dump(2).c_str());
    return result.ok ? 0 : 1;
}
