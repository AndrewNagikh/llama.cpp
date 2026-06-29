#include "verification/layer_equivalence.h"
#include "runtime_debug/layer_trace.h"

#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <string>

int main() {
    const char * mono   = std::getenv("LLAMA_TEST_MONO");
    const char * worker = std::getenv("LLAMA_TEST_WORKER");
    const char * prompt = std::getenv("LLAMA_TEST_PROMPT");

    if (mono == nullptr || worker == nullptr) {
        layer_equivalence_report stub{};
        stub.message = "skipped (set LLAMA_TEST_MONO and LLAMA_TEST_WORKER)";
        printf("test-layer-equivalence: SKIP %s\n", stub.message.c_str());
        return 0;
    }

    const std::string prompt_str = prompt ? prompt : "The capital of France is";
    const layer_trace_config cfg = layer_trace_load_config();
    const auto report            = verify_layer_equivalence(
            mono, worker, prompt_str, -1, &cfg);

    printf("%s\n", layer_equivalence_report_json(report).c_str());
    assert(report.layers_checked > 0);
    printf("test-layer-equivalence: checked=%d first_fail=%d\n",
            report.layers_checked,
            report.first_fail_layer);
    return 0;
}
