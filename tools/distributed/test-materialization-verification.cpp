#include "test_sync_common.h"
#include "test_verify_common.h"
#include "verification/verification_stages.h"

#include <cstdio>
#include <filesystem>

int main() {
    const std::string path = manifest_test_model_path();
    if (path.empty() || !std::filesystem::exists(path)) {
        fprintf(stderr, "test-materialization-verification: GGUF model not found (set MODEL env)\n");
        return 77;
    }

    model_manifest manifest;
    auto store = make_temp_layer_store("mat-verification");
    if (!verify_setup_store_from_model(path, store, manifest)) {
        fprintf(stderr, "test-materialization-verification: failed to populate layer store\n");
        return 1;
    }

    const std::string work = verify_test_work_dir("mat-verification");
    stage_pipeline_options options;
    options.stop_on_first_fail = true;
    options.run_sampling       = true;

    const auto report = run_materialization_verification_stages(
            "test-model",
            path,
            store,
            manifest,
            work,
            options);

    for (const auto & stage : report.stages) {
        printf("  %-35s %s\n",
                stage.name.c_str(),
                verify_status_to_string(stage.status).c_str());
        if (stage.status == verify_status::fail) {
            fprintf(stderr, "    %s\n", stage.message.c_str());
        }
    }

    if (!report.passed) {
        fprintf(stderr, "test-materialization-verification: FAIL at %s\n",
                report.failed_stage.c_str());
        return 1;
    }

    printf("test-materialization-verification: PASS\n");
    return 0;
}
