#include "orchestrator/install_planner/install_planner.h"
#include "test_install_common.h"

#include <cstdio>

int main() {
    const model_manifest manifest = make_manifest_with_layer_ranges(3, 10 * 1024 * 1024);
    const desired_model_layout desired = make_desired_layout("m", {
        { "a", 0 }, { "a", 1 }, { "b", 2 },
    });
    const actual_model_layout actual = make_actual_layout("m", {
        make_ready_layer(0, "a"),
    });
    const coverage_report coverage = make_coverage("m", { 1, 2 }, {}, 3);

    const auto result = build_install_plan(manifest, desired, actual, coverage);
    if (!result.success) {
        fprintf(stderr, "test-install-validation: build failed\n");
        return 1;
    }

    std::string err;
    if (!validate_install_plan(result.plan, desired, coverage, err)) {
        fprintf(stderr, "test-install-validation: %s\n", err.c_str());
        return 1;
    }

    const coverage_report ready_cov = make_coverage("m", {}, {}, 3);
    const auto ready_plan = build_install_plan(
            manifest, desired,
            make_actual_layout("m", {
                make_ready_layer(0, "a"),
                make_ready_layer(1, "a"),
                make_ready_layer(2, "b"),
            }),
            ready_cov);
    if (!ready_plan.success || ready_plan.plan.operation_count != 0) {
        fprintf(stderr, "test-install-validation: ready plan should be empty\n");
        return 1;
    }

    printf("test-install-validation: OK\n");
    return 0;
}
