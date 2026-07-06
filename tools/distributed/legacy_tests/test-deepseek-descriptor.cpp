#include "test_architecture_common.h"

#include <cstdio>

int main() {
    const bool ok = test_descriptor_from_gguf({
        "deepseek-r1-distill-qwen-1.5b-q4_k_m.gguf",
        "deepseek-coder-1.3b-q4_k_m.gguf",
    }, "deepseek");

    if (!ok) {
        printf("test-deepseek-descriptor: SKIP (no DeepSeek GGUF found)\n");
        return 77;
    }

    printf("test-deepseek-descriptor: OK\n");
    return 0;
}
