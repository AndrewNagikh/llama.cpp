#include "orchestrator/install_planner/install_planner.h"
#include "test_install_common.h"

#include <cstdio>

int main() {
    const model_manifest manifest = make_manifest_with_layer_ranges(8, 100 * 1024 * 1024);
    const desired_model_layout desired = make_desired_layout("m", {
        { "a", 0 }, { "a", 1 }, { "a", 2 }, { "b", 3 }, { "b", 4 },
    });
    const actual_model_layout actual = make_actual_layout("m", {
        make_ready_layer(0, "a"),
        make_ready_layer(1, "a"),
    });
    const coverage_report coverage = make_coverage("m", { 2, 3, 4 }, {}, 5);

    const auto result = build_install_plan(manifest, desired, actual, coverage, "file://model.gguf");
    if (!result.success) {
        fprintf(stderr, "test-install-plan: %s\n", result.error.c_str());
        return 1;
    }
    if (result.plan.operation_count < 3) {
        fprintf(stderr, "test-install-plan: expected at least 3 operations\n");
        return 1;
    }

    std::string err;
    if (!validate_install_plan(result.plan, desired, coverage, err)) {
        fprintf(stderr, "test-install-plan: validation failed: %s\n", err.c_str());
        return 1;
    }

    printf("test-install-plan: OK operations=%d bytes=%llu\n",
            result.plan.operation_count,
            (unsigned long long) result.plan.total_download_bytes);
    return 0;
}
