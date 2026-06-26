#include "test_architecture_common.h"

#include <cstdio>

int main() {
    const bool ok = test_descriptor_from_gguf({
        "gemma-3-1b-it-q4_k_m.gguf",
        "gemma-2-2b-it-q4_k_m.gguf",
    }, "gemma");

    if (!ok) {
        printf("test-gemma-descriptor: SKIP (no Gemma GGUF found)\n");
        return 77;
    }

    printf("test-gemma-descriptor: OK\n");
    return 0;
}
