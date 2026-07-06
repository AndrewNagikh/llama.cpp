#include "orchestrator/manifest_builder/manifest_builder.h"
#include "test_manifest_common.h"

#include <cassert>
#include <cstdio>
#include <filesystem>
#include <string>

int main() {
    const std::string path = manifest_test_model_path();
    if (path.empty() || !std::filesystem::exists(path)) {
        fprintf(stderr, "test-tensor-directory: GGUF model not found (set MODEL env)\n");
        return 77;
    }

    const model_manifest manifest = build_manifest_from_file(path);
    assert(!manifest.tensors.empty());

    bool has_blk0 = false;
    for (const auto & t : manifest.tensors) {
        assert(!t.name.empty());
        assert(!t.ggml_type.empty());
        assert(!t.dims.empty());
        assert(t.size_bytes > 0);
        assert(t.offset >= manifest.tensor_data_offset);
        if (t.name == "blk.0.attn_norm.weight") {
            has_blk0 = true;
            assert(t.layer == 0);
            assert(t.role == tensor_role::layer);
        }
    }
    assert(has_blk0);

    printf("test-tensor-directory: OK tensors=%zu\n", manifest.tensors.size());
    return 0;
}
