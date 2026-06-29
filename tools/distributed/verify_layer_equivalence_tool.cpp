#include "verification/layer_equivalence.h"
#include "runtime_debug/layer_trace.h"

#include <cstdio>
#include <cstdlib>
#include <string>

int main(int argc, char ** argv) {
    if (argc < 4) {
        fprintf(stderr,
                "usage: %s MONO_GGUF WORKER_ENTRY_GGUF PROMPT [MAX_LAYER]\n"
                "\n"
                "Env:\n"
                "  LLAMA_LAYER_TRACE=1       per-layer activation dumps\n"
                "  LLAMA_LAYER_TRACE_DIR=... output directory\n"
                "  LLAMA_LAYER_TRACE_INPUT=1 capture layer input (llama/qwen3 only)\n"
                "  LLAMA_LAYER_TRACE_OPS=1   dump single-block graph at fail layer\n"
                "  LLAMA_LAYER_TRACE_BLOCK=N restrict block trace to layer N\n"
                "  LLAMA_GRAPH_DUMP=1        dump ggml graph at first failure\n",
                argv[0]);
        return 2;
    }

    const std::string mono_path   = argv[1];
    const std::string worker_path = argv[2];
    const std::string prompt      = argv[3];
    const int32_t max_layer       = argc >= 5 ? std::stoi(argv[4]) : -1;

    const layer_trace_config cfg = layer_trace_load_config();
    const auto report          = verify_layer_equivalence(
            mono_path, worker_path, prompt, max_layer, &cfg);

    printf("%s\n", layer_equivalence_report_json(report).c_str());
    return report.all_pass ? 0 : 1;
}
