#include "test_sync_common.h"
#include "test_verify_common.h"
#include "verification/layer_verify.h"

#include <cstdio>
#include <filesystem>

int main() {
    const std::string path = manifest_test_model_path();
    if (path.empty() || !std::filesystem::exists(path)) {
        fprintf(stderr, "test-layer-checksum: GGUF model not found (set MODEL env)\n");
        return 77;
    }

    model_manifest manifest;
    auto store = make_temp_layer_store("layer-checksum");
    if (!verify_setup_store_from_model(path, store, manifest)) {
        fprintf(stderr, "test-layer-checksum: failed to populate layer store\n");
        return 1;
    }

    const auto result = layer_verify_store(store, manifest, path);
    if (result.summary.status != verify_status::ok) {
        fprintf(stderr, "test-layer-checksum: %s\n", result.summary.message.c_str());
        for (const auto & i : result.issues) {
            fprintf(stderr, "  %s\n", i.c_str());
        }
        return 1;
    }

    printf("test-layer-checksum: OK\n");
    return 0;
}
