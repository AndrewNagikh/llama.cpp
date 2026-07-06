#include "test_sync_common.h"
#include "test_verify_common.h"
#include "verification/tensor_verify.h"

#include <cstdio>
#include <filesystem>

int main() {
    const std::string path = manifest_test_model_path();
    if (path.empty() || !std::filesystem::exists(path)) {
        fprintf(stderr, "test-tensor-checksum: GGUF model not found (set MODEL env)\n");
        return 77;
    }

    model_manifest manifest;
    auto store = make_temp_layer_store("tensor-checksum");
    if (!verify_setup_store_from_model(path, store, manifest)) {
        fprintf(stderr, "test-tensor-checksum: failed to populate layer store\n");
        return 1;
    }

    const std::string work = verify_test_work_dir("tensor-checksum");
    const std::string mat  = verify_materialize_from_store(store, manifest, work);
    if (mat.empty()) {
        fprintf(stderr, "test-tensor-checksum: materialize failed\n");
        return 1;
    }

    const auto result = tensor_verify_files(path, mat);
    if (result.summary.status != verify_status::ok) {
        fprintf(stderr, "test-tensor-checksum: %s\n", result.summary.message.c_str());
        for (const auto & m : result.mismatches) {
            fprintf(stderr, "  %s\n", m.c_str());
        }
        return 1;
    }

    printf("test-tensor-checksum: OK tensors=%zu\n", result.tensors.size());
    return 0;
}
