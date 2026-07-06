#include "orchestrator/manifest_builder/manifest_builder.h"
#include "test_manifest_common.h"

#include "nlohmann/json.hpp"

#include <cassert>
#include <cstdio>
#include <filesystem>
#include <string>

int main() {
    const std::string path = manifest_test_model_path();
    if (path.empty() || !std::filesystem::exists(path)) {
        fprintf(stderr, "test-manifest-builder: GGUF model not found (set MODEL env)\n");
        return 77;
    }

    const model_manifest manifest = build_manifest_from_file(path);
    assert(!manifest.empty());
    assert(manifest.n_layer == 16 || manifest.n_layer > 0);
    assert(manifest.layers.size() == manifest.n_layer);
    assert(manifest.tensors.size() > manifest.n_layer);

    const nlohmann::json j = manifest.to_json();
    const model_manifest roundtrip = model_manifest::from_json(j);
    assert(roundtrip.architecture == manifest.architecture);
    assert(roundtrip.n_layer == manifest.n_layer);
    assert(roundtrip.tensors.size() == manifest.tensors.size());
    assert(roundtrip.layers.size() == manifest.layers.size());

    printf("test-manifest-builder: OK tensors=%zu layers=%u\n",
            manifest.tensors.size(), manifest.n_layer);
    return 0;
}
