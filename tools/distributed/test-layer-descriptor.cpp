#include "orchestrator/manifest_builder/manifest_builder.h"
#include "test_manifest_common.h"

#include <cassert>
#include <cstdio>
#include <filesystem>
#include <string>

int main() {
    const std::string path = manifest_test_model_path();
    if (path.empty() || !std::filesystem::exists(path)) {
        fprintf(stderr, "test-layer-descriptor: GGUF model not found (set MODEL env)\n");
        return 77;
    }

    const model_manifest manifest = build_manifest_from_file(path);
    assert(manifest.layers.size() == manifest.n_layer);

    uint64_t total_layer_bytes = 0;
    for (const auto & layer : manifest.layers) {
        assert(layer.layer_index >= 0);
        assert(layer.size_bytes > 0);
        assert(!layer.tensors.empty());
        total_layer_bytes += layer.size_bytes;
    }
    assert(total_layer_bytes > 0);

    printf("test-layer-descriptor: OK layers=%zu total_layer_bytes=%llu\n",
            manifest.layers.size(),
            (unsigned long long) total_layer_bytes);
    return 0;
}
