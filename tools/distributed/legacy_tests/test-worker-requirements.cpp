#include "test_architecture_common.h"

#include "architecture/tensor_plan.h"

#include <cassert>
#include <cstdio>

int main() {
    const auto tied   = build_architecture_descriptor(make_dense_manifest("llama", true));
    const auto untied = build_architecture_descriptor(make_dense_manifest("qwen2", false));

    const semantic_blob * tied_embedding = find_blob(tied.blobs, "embedding");
    assert(tied_embedding != nullptr);
    assert(tied_embedding->deploy == blob_deploy_target::all_nodes);

    const semantic_blob * untied_embedding = find_blob(untied.blobs, "embedding");
    assert(untied_embedding != nullptr);
    assert(untied_embedding->deploy == blob_deploy_target::entry_node);

    const auto emb  = make_tensor("token_embd.weight", tensor_role::embedding);
    const auto head = make_tensor("output.weight", tensor_role::lm_head);

    assert(descriptor_tensor_included(tied, emb, 2, 4, false, true));
    assert(!descriptor_tensor_included(untied, emb, 2, 4, false, true));
    assert(descriptor_tensor_included(untied, head, 2, 4, false, true));

    printf("test-worker-requirements: OK\n");
    return 0;
}
