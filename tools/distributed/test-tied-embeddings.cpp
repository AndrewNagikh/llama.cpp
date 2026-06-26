#include "test_architecture_common.h"

#include <cassert>
#include <cstdio>

int main() {
    const auto manifest = make_dense_manifest("llama", true, 8);
    const auto desc     = build_architecture_descriptor(manifest);

    assert(desc.tied_embeddings);
    assert(architecture_output_satisfied_by_embedding(desc));
    assert(architecture_materialize_needs_embedding_for_output(desc, false, true));
    assert(!architecture_materialize_needs_embedding_for_output(desc, true, true));

    const int32_t layer_start = 4;
    const int32_t layer_end   = 8;
    const bool include_embedding = false;
    const bool include_output    = true;

    int included = 0;
    for (const auto & t : manifest.tensors) {
        if (architecture_tensor_included(
                    desc, t, layer_start, layer_end, include_embedding, include_output)) {
            ++included;
        }
    }

    assert(included > 0);
    bool has_embedding = false;
    for (const auto & t : manifest.tensors) {
        if (t.role == tensor_role::embedding &&
                architecture_tensor_included(
                        desc, t, layer_start, layer_end, include_embedding, include_output)) {
            has_embedding = true;
        }
    }
    assert(has_embedding);

    printf("test-tied-embeddings: OK\n");
    return 0;
}
