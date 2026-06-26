#include "test_sync_common.h"
#include "test_verify_common.h"
#include "verification/gguf_diff.h"

#include <cstdio>
#include <filesystem>

int main() {
    const std::string path = manifest_test_model_path();
    if (path.empty() || !std::filesystem::exists(path)) {
        fprintf(stderr, "test-gguf-diff: GGUF model not found (set MODEL env)\n");
        return 77;
    }

    model_manifest manifest;
    auto store = make_temp_layer_store("gguf-diff");
    if (!verify_setup_store_from_model(path, store, manifest)) {
        fprintf(stderr, "test-gguf-diff: failed to populate layer store\n");
        return 1;
    }

    const std::string work = verify_test_work_dir("gguf-diff");
    const std::string mat  = verify_materialize_from_store(store, manifest, work);
    if (mat.empty()) {
        fprintf(stderr, "test-gguf-diff: materialize failed\n");
        return 1;
    }

    const auto diff = gguf_diff_files(path, mat);
    if (!diff.passed) {
        fprintf(stderr, "test-gguf-diff: differences found (%zu)\n", diff.differences.size());
        for (const auto & d : diff.differences) {
            fprintf(stderr, "  %s\n", d.c_str());
        }
        return 1;
    }

    printf("test-gguf-diff: OK\n");
    return 0;
}
