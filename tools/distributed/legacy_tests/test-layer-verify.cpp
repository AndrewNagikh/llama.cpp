#include "test_sync_common.h"

#include <cstdio>
#include <vector>

int main() {
    auto store = make_temp_layer_store("verify-test");
    std::vector<uint8_t> payload(256, 0x42);
    if (!store.store_layer(2, payload.data(), payload.size(), 0, 256, "manifest:layer:2")) {
        fprintf(stderr, "test-layer-verify: store failed\n");
        return 1;
    }

    if (!store.verify_layer(2, "manifest:layer:2")) {
        fprintf(stderr, "test-layer-verify: expected valid checksum\n");
        return 1;
    }

    if (store.verify_layer(2, "manifest:layer:999")) {
        fprintf(stderr, "test-layer-verify: expected checksum mismatch\n");
        return 1;
    }

    printf("test-layer-verify: OK\n");
    return 0;
}
