#include "test_architecture_common.h"

#include "node_agent/layer_store/layer_gguf_assembler.h"
#include "verification/layer_store_populate.h"
#include "verification/gguf_inspector.h"

#include <cassert>
#include <cstdio>
#include <filesystem>

int main() {
    const auto path = find_model_file({
        "qwen2.5-1.5b-instruct-q4_k_m.gguf",
        "qwen2.5-0.5b-instruct-q4_k_m.gguf",
    });
    if (!path.has_value()) {
        printf("test-qwen-materialization: SKIP (no Qwen GGUF)\n");
        return 77;
    }

    const model_manifest manifest = build_manifest_from_file(*path);
    assert(!manifest.empty());

    const auto desc = build_architecture_descriptor(manifest);
    assert(desc.family == "qwen");
    assert(desc.separate_lm_head);

    auto store = layer_store(std::filesystem::temp_directory_path() / "dist-qwen-mat-test", "qwen-mat");
    assert(populate_layer_store_from_gguf(store, manifest, *path));

    const std::string out = (std::filesystem::temp_directory_path() / "dist-qwen-mat-test/out.gguf").string();
    assert(layer_store_materialize_gguf(store, manifest, out, 0, (int32_t) manifest.n_layer, false, true));

    const auto info = gguf_inspect_file(out);
    assert(info.contains("tensors"));
    bool has_output = false;
    bool has_norm   = false;
    for (const auto & t : info["tensors"]) {
        const std::string name = t["name"].get<std::string>();
        if (name == "output.weight") {
            has_output = true;
        }
        if (name == "output_norm.weight") {
            has_norm = true;
        }
    }
    assert(has_output && has_norm);

    printf("test-qwen-materialization: OK\n");
    return 0;
}
