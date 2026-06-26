#include "test_architecture_common.h"

#include <cassert>
#include <cstdio>

int main() {
    const auto llama = make_dense_manifest("llama", true);
    const auto desc  = build_architecture_descriptor(llama);

    assert(desc.architecture == "llama");
    assert(desc.family == "llama");
    assert(desc.tied_embeddings);
    assert(!desc.separate_lm_head);
    assert(find_blob(desc.blobs, "output_norm") != nullptr);
    assert(find_blob(desc.blobs, "embedding") != nullptr);
    assert(!desc.blobs.empty());
    assert(!desc.worker_requirements.empty());

    const auto separate = make_dense_manifest("qwen2", false);
    const auto desc2    = build_architecture_descriptor(separate);
    assert(desc2.family == "qwen");
    assert(!desc2.tied_embeddings);
    assert(desc2.separate_lm_head);
    assert(find_blob(desc2.blobs, "output_head") != nullptr);

    assert(find_blob(desc2.blobs, "output_head") != nullptr);
    assert(find_blob(desc2.blobs, "output_norm") != nullptr);

    printf("test-architecture-descriptor: OK\n");
    return 0;
}
