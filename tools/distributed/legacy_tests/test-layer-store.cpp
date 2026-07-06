#include "test_sync_common.h"

#include <cstdio>
#include <vector>

int main() {
    auto store = make_temp_layer_store("llama-test");
    const model_manifest manifest = make_manifest_with_layer_ranges(4, 1024);
    if (!store.save_manifest(manifest)) {
        fprintf(stderr, "test-layer-store: save_manifest failed\n");
        return 1;
    }

    std::vector<uint8_t> payload(512, 0xAB);
    if (!store.store_layer(0, payload.data(), payload.size(), 100, 612, "manifest:layer:0")) {
        fprintf(stderr, "test-layer-store: store_layer failed\n");
        return 1;
    }

    if (!store.has_layer(0)) {
        fprintf(stderr, "test-layer-store: has_layer expected true\n");
        return 1;
    }

    std::vector<uint8_t> loaded;
    if (!store.load_layer(0, loaded) || loaded != payload) {
        fprintf(stderr, "test-layer-store: load_layer mismatch\n");
        return 1;
    }

    const auto layers = store.list_layers();
    if (layers.size() != 1 || layers[0].layer_index != 0) {
        fprintf(stderr, "test-layer-store: list_layers unexpected\n");
        return 1;
    }

    const auto restored = store.load_manifest();
    if (!restored.has_value() || restored->n_layer != 4) {
        fprintf(stderr, "test-layer-store: load_manifest failed\n");
        return 1;
    }

    printf("test-layer-store: OK\n");
    return 0;
}
