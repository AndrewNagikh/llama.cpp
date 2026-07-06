#include "test_sync_common.h"
#include "test_verify_common.h"
#include "verification/layer_verify.h"
#include "verification/verification_pipeline.h"

#include <cstdio>
#include <filesystem>

int main() {
    const std::string path = manifest_test_model_path();
    if (path.empty() || !std::filesystem::exists(path)) {
        fprintf(stderr, "test-layer-store-verification: GGUF model not found (set MODEL env)\n");
        return 77;
    }

    model_manifest manifest;
    auto store = make_temp_layer_store("layer-store-verification");
    if (!verify_setup_store_from_model(path, store, manifest)) {
        fprintf(stderr, "test-layer-store-verification: failed to populate layer store\n");
        return 1;
    }

    const auto layers = layer_verify_store(store, manifest, path);
    if (layers.summary.status != verify_status::ok) {
        fprintf(stderr, "test-layer-store-verification: layer verify failed\n");
        for (const auto & i : layers.issues) {
            fprintf(stderr, "  %s\n", i.c_str());
        }
        return 1;
    }

    const std::string work = verify_test_work_dir("layer-store-verification");
    const verification_report report = run_verification_pipeline(
            "layer-store-test", path, store, manifest, work);
    if (!report.passed) {
        fprintf(stderr, "test-layer-store-verification: pipeline failed\n");
        for (const auto & d : report.diffs) {
            fprintf(stderr, "  %s\n", d.c_str());
        }
        return 1;
    }

    printf("test-layer-store-verification: OK\n");
    return 0;
}
