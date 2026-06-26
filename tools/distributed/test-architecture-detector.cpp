#include "test_architecture_common.h"

#include "architecture/architecture_plugin.h"

#include <cassert>
#include <cstdio>

int main() {
    const auto llama_manifest = make_dense_manifest("llama", true);
    assert(select_architecture_plugin(llama_manifest).family() == "llama");

    const auto qwen_manifest = make_dense_manifest("qwen2", false);
    assert(select_architecture_plugin(qwen_manifest).family() == "qwen");

    const auto gemma_manifest = make_dense_manifest("gemma2", true);
    assert(select_architecture_plugin(gemma_manifest).family() == "gemma");

    printf("test-architecture-detector: OK\n");
    return 0;
}
