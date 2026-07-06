#include "verification/final_runtime_verify.h"

#include <cstdio>
#include <cstdlib>

int main() {
    const char * mono   = std::getenv("LLAMA_TEST_MONO_MODEL");
    const char * worker = std::getenv("LLAMA_TEST_WORKER_FINAL_MODEL");
    const char * prompt = std::getenv("LLAMA_TEST_PROMPT");
    if (mono == nullptr || worker == nullptr) {
        printf("test-final-runtime: SKIP (set LLAMA_TEST_MONO_MODEL and LLAMA_TEST_WORKER_FINAL_MODEL)\n");
        return 0;
    }

    final_runtime_config cfg{};
    cfg.mono_path         = mono;
    cfg.worker_final_path = worker;
    cfg.prompt            = prompt ? prompt : "The capital of France is";
    cfg.max_decode_steps  = 2;
    cfg.save_hidden_bin   = false;

    const auto report = verify_final_runtime(cfg);
    printf("%s\n", report.message.c_str());
    return report.all_pass ? 0 : 1;
}
