#include "test_architecture_common.h"

#include <cassert>
#include <cstdio>

int main() {
    const auto tied   = build_architecture_descriptor(make_dense_manifest("llama", true));
    const auto untied = build_architecture_descriptor(make_dense_manifest("qwen2", false));

    assert(architecture_worker_needs_embedding_blob(tied, worker_deploy_role::entry));
    assert(architecture_worker_needs_embedding_blob(tied, worker_deploy_role::final));
    assert(architecture_should_replicate_embedding(tied));

    assert(architecture_worker_needs_embedding_blob(untied, worker_deploy_role::entry));
    assert(!architecture_worker_needs_embedding_blob(untied, worker_deploy_role::final));
    assert(!architecture_should_replicate_embedding(untied));

    const auto emb  = make_tensor("token_embd.weight", tensor_role::embedding);
    const auto head = make_tensor("output.weight", tensor_role::lm_head);

    assert(architecture_tensor_included(tied, emb, 2, 4, false, true));
    assert(!architecture_tensor_included(untied, emb, 2, 4, false, true));
    assert(architecture_tensor_included(untied, head, 2, 4, false, true));

    printf("test-worker-requirements: OK\n");
    return 0;
}
