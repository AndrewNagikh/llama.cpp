#include "verification/runtime_api_verify.h"

#include <cstdio>
#include <cstdlib>

int main() {
    const char * model = std::getenv("LLAMA_TEST_MODEL");
    if (model == nullptr) {
        printf("test-runtime-api: SKIP (set LLAMA_TEST_MODEL)\n");
        return 0;
    }

    const auto result = verify_runtime_hidden_api(model, 1);
    printf("test-runtime-api: %s\n", result.message.c_str());
    return result.ok ? 0 : 1;
}
