#include "test_architecture_common.h"

#include "architecture/semantic_runtime_descriptor.h"

#include <cassert>
#include <cstdio>

int main() {
    const model_manifest manifest = make_dense_manifest("llama", false);
    const semantic_runtime_descriptor rt = build_semantic_runtime_descriptor(manifest);

    assert(rt.family == "llama");
    assert(!rt.blobs.empty());
    assert(!rt.workers.empty());
    assert(find_blob(rt.blobs, "embedding") != nullptr);
    assert(find_blob(rt.blobs, "output_head") != nullptr);
    assert(find_blob(rt.blobs, "output_norm") != nullptr);
    assert(find_blob(rt.blobs, "layer:0") != nullptr);

    const worker_materialize_plan entry_plan =
            worker_materialize_plan_for_role(rt, worker_role::entry, 0, 2);
    assert(!entry_plan.required_blobs.empty());

    const worker_materialize_plan final_plan =
            worker_materialize_plan_for_role(rt, worker_role::final, 2, 4);
    assert(!final_plan.required_blobs.empty());

    const worker_descriptor * entry_wd = find_worker_descriptor(rt, worker_role::entry);
    assert(entry_wd != nullptr);

    printf("test-runtime-descriptor: OK blobs=%zu workers=%zu\n",
            rt.blobs.size(), rt.workers.size());
    return 0;
}
