#include "verification/final_runtime_verify.h"

#include <cstdio>
#include <cstdlib>

int main() {
    const char * mono   = std::getenv("LLAMA_TEST_MONO_MODEL");
    const char * worker = std::getenv("LLAMA_TEST_WORKER_FINAL_MODEL");
    if (mono == nullptr || worker == nullptr) {
        printf("test-hidden-injection: SKIP\n");
        return 0;
    }

    final_runtime_config cfg{};
    cfg.mono_path         = mono;
    cfg.worker_final_path = worker;
    cfg.prompt            = "Hello";
    cfg.max_decode_steps  = 0;
    cfg.save_hidden_bin   = false;

    const auto report = verify_final_runtime(cfg);
    const bool ok     = !report.hidden_sha256.empty() && report.n_tokens > 0;
    printf("test-hidden-injection: %s hidden_sha256=%s n_tokens=%d\n",
            ok ? "OK" : "FAIL",
            report.hidden_sha256.substr(0, 12).c_str(),
            report.n_tokens);
    return ok ? 0 : 1;
}
