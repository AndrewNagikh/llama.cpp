#include "test_sync_common.h"
#include "test_verify_common.h"
#include "verification/alignment_verify.h"

#include <cstdio>
#include <filesystem>

int main() {
    const std::string path = manifest_test_model_path();
    if (path.empty() || !std::filesystem::exists(path)) {
        fprintf(stderr, "test-alignment: GGUF model not found (set MODEL env)\n");
        return 77;
    }

    model_manifest manifest;
    auto store = make_temp_layer_store("alignment");
    if (!verify_setup_store_from_model(path, store, manifest)) {
        fprintf(stderr, "test-alignment: failed to populate layer store\n");
        return 1;
    }

    const std::string work = verify_test_work_dir("alignment");
    const std::string mat  = verify_materialize_from_store(store, manifest, work);
    if (mat.empty()) {
        fprintf(stderr, "test-alignment: materialize failed\n");
        return 1;
    }

    const auto orig = alignment_verify_file(path);
    const auto mat_a = alignment_verify_file(mat);
    if (orig.summary.status != verify_status::ok) {
        fprintf(stderr, "test-alignment: original: %s\n", orig.summary.message.c_str());
        return 1;
    }
    if (mat_a.summary.status != verify_status::ok) {
        fprintf(stderr, "test-alignment: materialized: %s\n", mat_a.summary.message.c_str());
        for (const auto & i : mat_a.issues) {
            fprintf(stderr, "  %s\n", i.c_str());
        }
        return 1;
    }

    printf("test-alignment: OK\n");
    return 0;
}
