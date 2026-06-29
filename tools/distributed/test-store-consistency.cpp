#include "node_agent/layer_store/layer_store.h"
#include "orchestrator/consistency/store_verifier.h"
#include "test_install_common.h"
#include "test_sync_common.h"

#include <cstdio>
#include <vector>

int main() {
    auto store = make_temp_layer_store("store-consistency");
    const model_manifest manifest = make_manifest_with_layer_ranges(2, 512);
    if (!store.save_manifest(manifest)) {
        fprintf(stderr, "test-store-consistency: save_manifest failed\n");
        return 1;
    }

    std::vector<uint8_t> payload(512, 0x42);
    const std::string blob0 = "layer:0";
    const std::string blob1 = "layer:1";
    const std::string tensor0 = manifest.tensors[0].name;
    const std::string tensor1 = manifest.tensors[1].name;
    const std::string checksum0 = "manifest:tensor:" + tensor0;
    const std::string checksum1 = "manifest:tensor:" + tensor1;

    if (!store.store_blob_tensor(blob0, tensor0, payload.data(), payload.size(), 100, checksum0)) {
        fprintf(stderr, "test-store-consistency: store_blob_tensor(0) failed\n");
        return 1;
    }

    store_verify_result verified = verify_layer_store(store, manifest, "node-a");
    if (verified.missing_count != 1) {
        fprintf(stderr, "test-store-consistency: expected one missing layer blob got missing=%d\n",
                verified.missing_count);
        return 1;
    }

    if (!store.store_blob_tensor(blob1, tensor1, payload.data(), payload.size(), 700, checksum1)) {
        fprintf(stderr, "test-store-consistency: store_blob_tensor(1) failed\n");
        return 1;
    }

    verified = verify_layer_store(store, manifest, "node-a");
    if (!verified.ok || verified.verified_count == 0) {
        fprintf(stderr, "test-store-consistency: verify_layer_store failed after full store\n");
        return 1;
    }

    actual_model_layout actual = make_actual_layout("store-consistency", {
        make_ready_layer(0, "node-a", checksum0),
        make_ready_layer(1, "node-a", checksum1),
    });
    actual.layers[0].blob_id     = blob0;
    actual.layers[0].tensor_name = tensor0;
    actual.layers[1].blob_id     = blob1;
    actual.layers[1].tensor_name = tensor1;

    const store_verify_result compared =
            compare_store_to_actual(verified, actual, "node-a");
    if (!compared.ok) {
        fprintf(stderr, "test-store-consistency: store/registry mismatch\n");
        for (const auto & issue : compared.issues) {
            fprintf(stderr, "  %s\n", issue.c_str());
        }
        return 1;
    }

    if (!store.clear_model_storage(true)) {
        fprintf(stderr, "test-store-consistency: clear_model_storage failed\n");
        return 1;
    }
    if (!store.list_blob_tensors().empty()) {
        fprintf(stderr, "test-store-consistency: blob tensors not cleared\n");
        return 1;
    }
    if (!store.load_manifest().has_value()) {
        fprintf(stderr, "test-store-consistency: manifest should remain after reset\n");
        return 1;
    }

    verified = verify_layer_store(store, manifest, "node-a");
    if (verified.ok || verified.missing_count == 0) {
        fprintf(stderr, "test-store-consistency: expected missing blobs after reset\n");
        return 1;
    }

    printf("test-store-consistency: OK\n");
    return 0;
}
