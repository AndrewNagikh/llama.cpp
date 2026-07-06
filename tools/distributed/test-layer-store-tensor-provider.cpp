#include "architecture/semantic_blob.h"
#include "architecture/semantic_runtime_descriptor.h"
#include "runtime/layer_store_tensor_provider.h"
#include "test_sync_common.h"

#include <cstdio>
#include <vector>

static int test_tensor_exists_after_store() {
    auto store = make_temp_layer_store("tensor-provider");
    const model_manifest manifest = make_manifest_with_layer_ranges(2, 512);
    if (!store.save_manifest(manifest)) {
        fprintf(stderr, "test-layer-store-tensor-provider: save_manifest failed\n");
        return 1;
    }

    const semantic_runtime_descriptor rt = build_semantic_runtime_descriptor(manifest);
    const std::string tensor_name = "blk.0.attn.weight";
    layer_store_tensor_provider provider(store, rt);
    if (provider.tensor_exists(tensor_name)) {
        fprintf(stderr, "test-layer-store-tensor-provider: tensor should not exist before store\n");
        return 1;
    }

    std::vector<uint8_t> payload(512, 0xCD);
    if (!store.store_blob_tensor(
                layer_blob_id(0),
                tensor_name,
                payload.data(),
                payload.size(),
                100,
                "manifest:layer:0")) {
        fprintf(stderr, "test-layer-store-tensor-provider: store_blob_tensor failed\n");
        return 1;
    }

    layer_store_tensor_provider provider2(store, rt);
    if (!provider2.tensor_exists(tensor_name)) {
        fprintf(stderr, "test-layer-store-tensor-provider: tensor_exists expected true\n");
        return 1;
    }

    std::vector<uint8_t> loaded;
    if (!provider2.load_tensor(tensor_name, loaded) || loaded.size() != payload.size()) {
        fprintf(stderr, "test-layer-store-tensor-provider: load_tensor failed\n");
        return 1;
    }
    return 0;
}

int main() {
    if (test_tensor_exists_after_store() != 0) {
        fprintf(stderr, "test-layer-store-tensor-provider: FAILED\n");
        return 1;
    }
    printf("test-layer-store-tensor-provider: OK\n");
    return 0;
}
