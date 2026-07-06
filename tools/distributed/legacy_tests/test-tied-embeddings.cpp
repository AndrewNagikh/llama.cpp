#include "test_architecture_common.h"

#include "architecture/tensor_plan.h"

#include <cassert>
#include <cstdio>

int main() {
    const auto manifest = make_dense_manifest("llama", true, 8);
    const auto desc     = build_architecture_descriptor(manifest);

    assert(desc.tied_embeddings);
    const semantic_blob * output_head = find_blob(desc.blobs, "output_head");
    assert(output_head != nullptr);
    assert(output_head->storage_alias);
    assert(output_head->storage_blob_id == "embedding");

    const int32_t layer_start = 4;
    const int32_t layer_end   = 8;
    const bool include_embedding = false;
    const bool include_output    = true;

    bool has_embedding = false;
    for (const auto & t : manifest.tensors) {
        if (t.role == tensor_role::embedding &&
                descriptor_tensor_included(
                        desc, t, layer_start, layer_end, include_embedding, include_output)) {
            has_embedding = true;
        }
    }
    assert(has_embedding);

    printf("test-tied-embeddings: OK\n");
    return 0;
}
