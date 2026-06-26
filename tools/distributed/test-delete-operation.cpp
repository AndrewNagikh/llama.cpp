#include "orchestrator/install_planner/install_planner.h"
#include "test_install_common.h"

#include <cstdio>

int main() {
    const model_manifest manifest = make_manifest_with_layer_ranges(4, 30 * 1024 * 1024);
    const desired_model_layout desired = make_desired_layout("m", { { "a", 2 } });
    const actual_model_layout actual = make_actual_layout("m", {
        make_ready_layer(2, "b"),
    });
    const coverage_report coverage = make_coverage("m", { 2 }, {}, 1);

    const auto result = build_install_plan(manifest, desired, actual, coverage);
    if (!result.success) {
        fprintf(stderr, "test-delete-operation: %s\n", result.error.c_str());
        return 1;
    }

    bool has_delete = false;
    bool has_download = false;
    for (const auto & op : result.plan.operations) {
        if (op.action == install_action::delete_op && op.node_id == "b" && op.layer_index == 2) {
            has_delete = true;
        }
        if (op.action == install_action::download && op.node_id == "a" && op.layer_index == 2) {
            has_download = true;
        }
    }
    if (!has_delete || !has_download) {
        fprintf(stderr, "test-delete-operation: expected DELETE on b and DOWNLOAD on a\n");
        return 1;
    }

    printf("test-delete-operation: OK\n");
    return 0;
}
