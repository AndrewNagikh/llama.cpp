#include "test_manifest_common.h"
#include "test_sync_common.h"
#include "test_verify_common.h"
#include "verification/verification_stages.h"

#include <cstdio>
#include <filesystem>

int main(int argc, char ** argv) {
    std::string model_path = manifest_test_model_path();
    if (argc >= 2) {
        model_path = argv[1];
    }
    if (model_path.empty() || !std::filesystem::exists(model_path)) {
        fprintf(stderr, "usage: %s [MODEL.gguf]\n", argv[0]);
        return 77;
    }

    model_manifest manifest;
    auto store = make_temp_layer_store("materialization-verify");
    if (!verify_setup_store_from_model(model_path, store, manifest)) {
        fprintf(stderr, "materialization_verify: failed to populate layer store\n");
        return 1;
    }

    const std::string work = verify_test_work_dir("materialization-verify");
    stage_pipeline_options options;
    options.stop_on_first_fail = false;
    options.run_sampling       = true;

    const auto report = run_materialization_verification_stages(
            "verify-model",
            model_path,
            store,
            manifest,
            work,
            options);

    printf("%s\n", report.to_json().dump(2).c_str());
    if (!report.passed) {
        fprintf(stderr, "materialization_verify: FAIL at %s\n", report.failed_stage.c_str());
        return 1;
    }
    fprintf(stderr, "materialization_verify: PASS\n");
    return 0;
}
