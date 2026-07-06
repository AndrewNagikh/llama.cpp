#include "test_sync_common.h"
#include "test_verify_common.h"
#include "verification/materialization_verify.h"

#include <cstdio>
#include <filesystem>

int main() {
    const std::string path = manifest_test_model_path();
    if (path.empty() || !std::filesystem::exists(path)) {
        fprintf(stderr, "test-materialization-repeatability: GGUF model not found (set MODEL env)\n");
        return 77;
    }

    model_manifest manifest;
    auto store = make_temp_layer_store("repeatability");
    if (!verify_setup_store_from_model(path, store, manifest)) {
        fprintf(stderr, "test-materialization-repeatability: failed to populate layer store\n");
        return 1;
    }

    const std::string work = verify_test_work_dir("repeatability");
    const auto result = verify_materialization_repeatability(store, manifest, work);
    if (result.summary.status != verify_status::ok) {
        fprintf(stderr, "test-materialization-repeatability: %s\n", result.summary.message.c_str());
        return 1;
    }

    printf("test-materialization-repeatability: OK sha256=%s\n", result.sha256_repeat.c_str());
    return 0;
}
