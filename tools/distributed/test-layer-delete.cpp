#include "test_sync_common.h"

#include <cstdio>
#include <vector>

int main() {
    auto store = make_temp_layer_store("layer-delete");
    std::vector<uint8_t> payload(128, 0x55);
    if (!store.store_layer(3, payload.data(), payload.size(), 0, 128, "manifest:layer:3")) {
        fprintf(stderr, "test-layer-delete: store failed\n");
        return 1;
    }

    if (!store.remove_layer(3) || store.has_layer(3)) {
        fprintf(stderr, "test-layer-delete: remove_layer failed\n");
        return 1;
    }

    if (!store.list_layers().empty()) {
        fprintf(stderr, "test-layer-delete: list_layers not empty\n");
        return 1;
    }

    printf("test-layer-delete: OK\n");
    return 0;
}
