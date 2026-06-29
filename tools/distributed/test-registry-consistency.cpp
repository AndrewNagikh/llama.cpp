#include "orchestrator/consistency/cluster_consistency.h"
#include "orchestrator/install_planner/install_planner.h"
#include "orchestrator/model_registry.h"
#include "test_install_common.h"

#include <cstdio>
#include <set>

int main() {
    cluster_model_registry reg;
    dist_model_record record;
    record.model_id = "registry-m";
    reg.add_or_update(record);

    const model_manifest manifest = make_manifest_with_layer_ranges(3, 8 * 1024 * 1024);
    reg.apply_manifest("registry-m", manifest, nullptr);

    const desired_model_layout desired = make_desired_layout("registry-m", {
        { "a", 0 }, { "b", 1 }, { "b", 2 },
    });
    reg.apply_layout("registry-m", desired, nullptr);

    const actual_model_layout actual = make_actual_layout("registry-m", {
        make_ready_layer(0, "a"),
        make_ready_layer(1, "b"),
    });
    reg.apply_actual("registry-m", actual, nullptr);
    reg.apply_coverage("registry-m", make_coverage("registry-m", { 2 }, {}, 3), nullptr);

    install_plan plan;
    plan.operation_count = 1;
    reg.apply_install_plan("registry-m", plan, nullptr);

    dist_model_record * found = reg.find("registry-m");
    if (!found || !found->actual.has_value() || !found->coverage.has_value() ||
            !found->stored_install_plan.has_value()) {
        fprintf(stderr, "test-registry-consistency: registry state not populated\n");
        return 1;
    }

    const std::set<std::string> online = { "a", "b" };
    consistency_check_result partial = check_cluster_consistency(
            *found, actual, online, "file://model.gguf");
    if (partial.idempotent) {
        fprintf(stderr, "test-registry-consistency: partial install should not be idempotent\n");
        return 1;
    }

    if (!reg.clear_install_cluster_state("registry-m", found)) {
        fprintf(stderr, "test-registry-consistency: clear_install_cluster_state failed\n");
        return 1;
    }

    found = reg.find("registry-m");
    if (!found || found->actual.has_value() || found->coverage.has_value() ||
            found->stored_install_plan.has_value()) {
        fprintf(stderr, "test-registry-consistency: install cluster state not cleared\n");
        return 1;
    }
    if (!found->layout.has_value() || !found->manifest.has_value()) {
        fprintf(stderr, "test-registry-consistency: layout/manifest should remain\n");
        return 1;
    }

    actual_model_layout empty_actual = make_actual_layout("registry-m", {});
    consistency_check_result cleared = check_cluster_consistency(
            *found, empty_actual, online, "file://model.gguf");
    if (cleared.idempotent) {
        fprintf(stderr, "test-registry-consistency: empty cluster should need install ops\n");
        return 1;
    }
    if (cleared.rebuilt_plan.operation_count == 0) {
        fprintf(stderr, "test-registry-consistency: expected non-empty plan after reset\n");
        return 1;
    }

    printf("test-registry-consistency: OK\n");
    return 0;
}
