#include "test_architecture_common.h"

#include "architecture/semantic_runtime_descriptor.h"
#include "node_agent/layer_store/layer_store.h"
#include "node_agent/layer_store/worker_builder.h"

#include <cassert>
#include <cstdio>
#include <filesystem>

int main() {
    const model_manifest manifest = make_dense_manifest("llama", false, 2);
    const semantic_runtime_descriptor rt = build_semantic_runtime_descriptor(manifest);

    const auto tmp = std::filesystem::temp_directory_path() / "test-worker-builder";
    std::error_code ec;
    std::filesystem::remove_all(tmp, ec);

    layer_store store(tmp, "test-model");
    assert(store.save_manifest(manifest));

    const std::vector<uint8_t> meta(manifest.tensor_data_offset, 0xAB);
    assert(store.save_metadata_blob(meta));

    for (int32_t layer = 0; layer < 2; ++layer) {
        const std::vector<uint8_t> layer_data(1024, static_cast<uint8_t>(layer + 1));
        assert(store.store_layer(layer, layer_data.data(), layer_data.size(), 5000, 6024, ""));
    }

    const std::string out_path = (store.model_root() / "worker_middle.gguf").string();
    std::string err;
    assert(materialize_worker_gguf(
            store, manifest, rt, worker_role::middle, 0, 2, out_path, err));

    assert(std::filesystem::exists(out_path));
    assert(std::filesystem::file_size(out_path) >= manifest.tensor_data_offset);

    std::filesystem::remove_all(tmp, ec);
    printf("test-worker-builder: OK\n");
    return 0;
}
