#include "test_architecture_common.h"

#include <cassert>
#include <cstdio>

int main() {
    const auto manifest = make_dense_manifest("qwen2", false, 8);
    const auto desc     = build_architecture_descriptor(manifest);

    assert(!desc.tied_embeddings);
    assert(desc.has_separate_output);
    assert(!architecture_should_replicate_embedding(desc));

    const int32_t layer_start = 4;
    const int32_t layer_end   = 8;
    const bool include_embedding = false;
    const bool include_output    = true;

    bool has_lm_head = false;
    bool has_embedding = false;
    for (const auto & t : manifest.tensors) {
        const bool inc = architecture_tensor_included(
                desc, t, layer_start, layer_end, include_embedding, include_output);
        if (t.role == tensor_role::lm_head && inc) {
            has_lm_head = true;
        }
        if (t.role == tensor_role::embedding && inc) {
            has_embedding = true;
        }
    }

    assert(has_lm_head);
    assert(!has_embedding);

    printf("test-output-head: OK\n");
    return 0;
}
