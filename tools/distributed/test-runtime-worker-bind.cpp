#include "runtime/runtime_worker_bind.h"
#include "test_sync_common.h"

#include <cstdio>

static int test_sampler_bind() {
    auto store = make_temp_layer_store("bind-sampler");
    model_manifest manifest;
    manifest.n_layer = 4;
    if (!store.save_manifest(manifest)) {
        fprintf(stderr, "test-runtime-worker-bind: save_manifest failed\n");
        return 1;
    }

    const runtime_bind_result bind = runtime_bind_worker(
            store, manifest, worker_role::sampler, 0, 0);
    if (!bind.success || !bind.tensors_ready || bind.materialize_required) {
        fprintf(stderr, "test-runtime-worker-bind: sampler bind unexpected\n");
        return 1;
    }
    return 0;
}

static int test_missing_layer_tensors() {
    auto store = make_temp_layer_store("bind-missing");
    const model_manifest manifest = make_manifest_with_layer_ranges(4, 1024);
    if (!store.save_manifest(manifest)) {
        fprintf(stderr, "test-runtime-worker-bind: save_manifest failed\n");
        return 1;
    }

    const runtime_bind_result bind = runtime_bind_worker(
            store, manifest, worker_role::pipeline_stage, 0, 2);
    if (bind.tensors_ready) {
        fprintf(stderr, "test-runtime-worker-bind: expected missing layer tensors\n");
        return 1;
    }
    if (bind.error.empty()) {
        fprintf(stderr, "test-runtime-worker-bind: expected bind error\n");
        return 1;
    }
    return 0;
}

int main() {
    if (test_sampler_bind() != 0 || test_missing_layer_tensors() != 0) {
        fprintf(stderr, "test-runtime-worker-bind: FAILED\n");
        return 1;
    }
    printf("test-runtime-worker-bind: OK\n");
    return 0;
}
