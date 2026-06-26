#include "orchestrator/install_planner/install_planner.h"
#include "orchestrator/model_registry.h"
#include "test_install_common.h"

#include <cstdio>

int main() {
    cluster_model_registry reg;
    dist_model_record record;
    record.model_id = "m";
    reg.add_or_update(record);

    const model_manifest manifest = make_manifest_with_layer_ranges(6, 25 * 1024 * 1024);
    reg.apply_manifest("m", manifest, nullptr);

    const desired_model_layout desired = make_desired_layout("m", {
        { "a", 0 }, { "a", 1 }, { "b", 2 }, { "b", 3 },
    });
    reg.apply_layout("m", desired, nullptr);

    const actual_model_layout actual = make_actual_layout("m", {
        make_ready_layer(0, "a"),
    });
    reg.apply_actual("m", actual, nullptr);

    const coverage_report coverage = make_coverage("m", { 1, 2, 3 }, {}, 4);
    reg.apply_coverage("m", coverage, nullptr);

    const auto * stored = reg.find("m");
    const auto built = build_install_plan(
            *stored->manifest,
            stored->layout->desired,
            *stored->actual,
            *stored->coverage,
            "file://model.gguf");
    if (!built.success) {
        fprintf(stderr, "test-build-install-plan: %s\n", built.error.c_str());
        return 1;
    }
    reg.apply_install_plan("m", built.plan, nullptr);

    if (!reg.find("m")->install_plan.has_value()) {
        fprintf(stderr, "test-build-install-plan: plan not stored\n");
        return 1;
    }

    printf("test-build-install-plan: OK operations=%d\n", built.plan.operation_count);
    return 0;
}
