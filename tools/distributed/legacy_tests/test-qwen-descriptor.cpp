#include "test_architecture_common.h"

#include <cstdio>

int main() {
    const bool ok = test_descriptor_from_gguf({
        "qwen2.5-1.5b-instruct-q4_k_m.gguf",
        "qwen2.5-0.5b-instruct-q4_k_m.gguf",
    }, "qwen");

    if (!ok) {
        printf("test-qwen-descriptor: SKIP (no Qwen GGUF found)\n");
        return 77;
    }

    printf("test-qwen-descriptor: OK\n");
    return 0;
}
