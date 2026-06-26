#include "orchestrator/install_planner/install_planner.h"
#include "test_install_common.h"

#include <cstdio>

int main() {
    const model_manifest manifest = make_manifest_with_layer_ranges(4, 40 * 1024 * 1024);
    const desired_model_layout desired = make_desired_layout("m", { { "a", 1 } });

    installed_layer corrupted = make_ready_layer(1, "a");
    corrupted.state = install_state::corrupted;
    const actual_model_layout actual = make_actual_layout("m", { corrupted });
    const coverage_report coverage = make_coverage("m", {}, { 1 }, 1);

    const auto result = build_install_plan(manifest, desired, actual, coverage);
    if (!result.success || result.plan.operations.empty()) {
        fprintf(stderr, "test-repair-operation: build failed\n");
        return 1;
    }
    if (result.plan.operations.front().action != install_action::repair) {
        fprintf(stderr, "test-repair-operation: expected REPAIR\n");
        return 1;
    }

    printf("test-repair-operation: OK\n");
    return 0;
}
