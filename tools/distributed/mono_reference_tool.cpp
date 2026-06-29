#include "verification/monolithic_trace.h"

#include "runtime_debug/runtime_debug.h"

#include <cstdio>
#include <cstdlib>
#include <string>

int main(int argc, char ** argv) {
    if (argc < 3) {
        fprintf(stderr, "usage: %s MODEL.gguf PROMPT [max_tokens] [session_id]\n", argv[0]);
        return 2;
    }
    setenv("LLAMA_DISTRIBUTED_DEBUG", "1", 1);
    if (const char * dir = std::getenv("LLAMA_DIST_TRACE_DIR")) {
        (void) dir;
    }

    const std::string model  = argv[1];
    const std::string prompt = argv[2];
    const int max_tokens     = argc >= 4 ? std::stoi(argv[3]) : 16;
    const std::string session = argc >= 5 ? argv[4] : "mono_ref";

    const mono_trace_result result = run_monolithic_trace(model, prompt, max_tokens, session);
    if (!result.ok) {
        fprintf(stderr, "mono_reference failed: %s\n", result.error.c_str());
        return 1;
    }
    printf("trace: %s\n", result.trace_path.c_str());
    printf("steps: %zu\n", result.steps.size());
    return 0;
}
