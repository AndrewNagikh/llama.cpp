#include "orchestrator/manifest_builder/manifest_builder.h"
#include "test_manifest_common.h"

#include <cassert>
#include <cstdio>
#include <filesystem>
#include <string>

int main() {
    const std::string path = manifest_test_model_path();
    if (path.empty() || !std::filesystem::exists(path)) {
        fprintf(stderr, "test-gguf-metadata: GGUF model not found (set MODEL env)\n");
        return 77;
    }

    const model_manifest manifest = build_manifest_from_file(path);

    assert(!manifest.architecture.empty());
    assert(manifest.n_layer > 0);
    assert(manifest.n_ctx > 0);
    assert(manifest.n_vocab > 0);
    assert(manifest.n_embd > 0);
    assert(manifest.tensor_data_offset > 0);
    assert(manifest.metadata_bytes_read > 0);
    assert(manifest.metadata_bytes_read < (uint64_t) std::filesystem::file_size(path));
    assert(manifest.special_tensors.count("embedding") > 0);
    assert(manifest.special_tensors.count("lm_head") > 0);

    printf("test-gguf-metadata: OK arch=%s n_layer=%u n_ctx=%u meta_bytes=%llu\n",
            manifest.architecture.c_str(),
            manifest.n_layer,
            manifest.n_ctx,
            (unsigned long long) manifest.metadata_bytes_read);
    return 0;
}
