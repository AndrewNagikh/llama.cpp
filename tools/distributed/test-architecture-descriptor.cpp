#include "test_architecture_common.h"

#include <cassert>
#include <cstdio>

int main() {
    const auto llama = make_dense_manifest("llama", true);
    const auto desc  = build_architecture_descriptor(llama);

    assert(desc.architecture == "llama");
    assert(desc.tied_embeddings);
    assert(!desc.has_separate_output);
    assert(desc.has_output_norm);
    assert(desc.tensors.size() == llama.tensors.size());
    assert(!desc.role_requirements.empty());

    const auto separate = make_dense_manifest("qwen2", false);
    const auto desc2    = build_architecture_descriptor(separate);
    assert(!desc2.tied_embeddings);
    assert(desc2.has_separate_output);

    printf("test-architecture-descriptor: OK\n");
    return 0;
}
