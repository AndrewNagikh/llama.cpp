#include "orchestrator/install_planner/install_planner.h"
#include "test_install_common.h"

#include <cstdio>

int main() {
    const model_manifest manifest = make_manifest_with_layer_ranges(10, 20 * 1024 * 1024);
    const desired_model_layout desired = make_desired_layout("m", {
        { "a", 5 }, { "a", 6 }, { "a", 7 },
    });
    const actual_model_layout actual = make_actual_layout("m", {});
    const coverage_report coverage = make_coverage("m", { 5, 6, 7 }, {}, 3);

    const auto result = build_install_plan(manifest, desired, actual, coverage);
    if (!result.success) {
        fprintf(stderr, "test-operation-grouping: %s\n", result.error.c_str());
        return 1;
    }
    if (!install_plan_has_grouped_layers(result.plan, "a", install_action::download, { 5, 6, 7 })) {
        fprintf(stderr, "test-operation-grouping: expected grouped layers 5,6,7 on node a\n");
        return 1;
    }

    printf("test-operation-grouping: OK\n");
    return 0;
}
