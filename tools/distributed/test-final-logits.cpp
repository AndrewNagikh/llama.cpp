#include "verification/final_runtime_verify.h"

#include <cstdio>
#include <cstdlib>

int main() {
    const char * mono   = std::getenv("LLAMA_TEST_MONO_MODEL");
    const char * worker = std::getenv("LLAMA_TEST_WORKER_FINAL_MODEL");
    if (mono == nullptr || worker == nullptr) {
        printf("test-final-logits: SKIP\n");
        return 0;
    }

    final_runtime_config cfg{};
    cfg.mono_path         = mono;
    cfg.worker_final_path = worker;
    cfg.prompt            = "The capital of France is";
    cfg.max_decode_steps  = 0;
    cfg.save_hidden_bin   = false;

    const auto report = verify_final_runtime(cfg);
    printf("test-final-logits: prefill_pass=%d argmax mono=%d worker=%d\n",
            report.prefill_logits_pass ? 1 : 0,
            report.mono_prefill_logits.argmax,
            report.worker_prefill_logits.argmax);
    return 0;
}
