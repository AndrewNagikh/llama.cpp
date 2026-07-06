#include "test_architecture_common.h"

#include "architecture/tensor_plan.h"

#include <cassert>
#include <cstdio>

int main() {
    const auto manifest = make_dense_manifest("qwen2", false, 8);
    const auto desc     = build_architecture_descriptor(manifest);

    assert(!desc.tied_embeddings);
    assert(desc.separate_lm_head);

    const semantic_blob * head_blob = find_blob(desc.blobs, "output_head");
    assert(head_blob != nullptr);
    assert(!head_blob->storage_alias);
    assert(head_blob->deploy == blob_deploy_target::final_node);

    const int32_t layer_start = 4;
    const int32_t layer_end   = 8;
    const bool include_embedding = false;
    const bool include_output    = true;

    bool has_lm_head = false;
    bool has_embedding = false;
    for (const auto & t : manifest.tensors) {
        const bool inc = descriptor_tensor_included(
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
