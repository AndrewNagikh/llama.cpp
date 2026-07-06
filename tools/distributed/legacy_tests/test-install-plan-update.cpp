#include "orchestrator/install_planner/install_planner.h"
#include "orchestrator/model_registry.h"
#include "test_install_common.h"

#include <cstdio>

int main() {
    cluster_model_registry reg;
    dist_model_record record;
    record.model_id = "m";
    reg.add_or_update(record);

    const model_manifest manifest = make_manifest_with_layer_ranges(4, 15 * 1024 * 1024);
    reg.apply_manifest("m", manifest, nullptr);

    const desired_model_layout desired = make_desired_layout("m", {
        { "a", 0 }, { "a", 1 },
    });
    reg.apply_layout("m", desired, nullptr);

    const actual_model_layout partial = make_actual_layout("m", {
        make_ready_layer(0, "a"),
    });
    reg.apply_actual("m", partial, nullptr);
    reg.apply_coverage("m", make_coverage("m", { 1 }, {}, 2), nullptr);

    const auto first = build_install_plan(
            manifest, desired, partial, *reg.find("m")->coverage);
    if (!first.success || first.plan.operation_count != 1) {
        fprintf(stderr, "test-install-plan-update: expected one operation initially\n");
        return 1;
    }

    const actual_model_layout full = make_actual_layout("m", {
        make_ready_layer(0, "a"),
        make_ready_layer(1, "a"),
    });
    reg.apply_actual("m", full, nullptr);
    reg.apply_coverage("m", make_coverage("m", {}, {}, 2), nullptr);

    const auto second = build_install_plan(
            manifest, desired, full, *reg.find("m")->coverage);
    if (!second.success || second.plan.operation_count != 0) {
        fprintf(stderr, "test-install-plan-update: expected empty plan after coverage update\n");
        return 1;
    }

    printf("test-install-plan-update: OK\n");
    return 0;
}
