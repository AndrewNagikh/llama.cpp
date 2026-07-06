#include "orchestrator/consistency/cluster_consistency.h"
#include "orchestrator/install_planner/install_planner.h"
#include "orchestrator/model_registry.h"
#include "test_install_common.h"

#include <cstdio>
#include <set>

int main() {
    cluster_model_registry reg;
    dist_model_record record;
    record.model_id = "idempotent-m";
    reg.add_or_update(record);

    const model_manifest manifest = make_manifest_with_layer_ranges(4, 15 * 1024 * 1024);
    reg.apply_manifest("idempotent-m", manifest, nullptr);

    const desired_model_layout desired = make_desired_layout("idempotent-m", {
        { "a", 0 }, { "a", 1 }, { "b", 2 }, { "b", 3 },
    });
    reg.apply_layout("idempotent-m", desired, nullptr);

    const actual_model_layout actual = make_actual_layout("idempotent-m", {
        make_ready_layer(0, "a"),
        make_ready_layer(1, "a"),
        make_ready_layer(2, "b"),
        make_ready_layer(3, "b"),
    });
    reg.apply_actual("idempotent-m", actual, nullptr);
    reg.apply_coverage("idempotent-m", make_coverage("idempotent-m", {}, {}, 4), nullptr);

    record = *reg.find("idempotent-m");
    record.stored_install_plan = install_plan{};

    const auto first = build_install_plan(
            manifest, desired, actual, *reg.find("idempotent-m")->coverage, "file://model.gguf");
    if (!first.success) {
        fprintf(stderr, "test-install-idempotency: first plan failed: %s\n", first.error.c_str());
        return 1;
    }
    if (first.plan.operation_count != 0) {
        fprintf(stderr, "test-install-idempotency: expected empty first plan got %d ops\n",
                first.plan.operation_count);
        return 1;
    }

    const auto second = build_install_plan(
            manifest, desired, actual, *reg.find("idempotent-m")->coverage, "file://model.gguf");
    if (!second.success || second.plan.operation_count != 0) {
        fprintf(stderr, "test-install-idempotency: expected empty second plan\n");
        return 1;
    }

    const install_plan_diff diff = diff_install_plan(first.plan, second.plan);
    if (diff.added_ops != 0 || !diff.operations.empty()) {
        fprintf(stderr, "test-install-idempotency: unexpected plan diff\n");
        return 1;
    }

    const std::set<std::string> online = { "a", "b" };
    const consistency_check_result check = check_cluster_consistency(
            record, actual, online, "file://model.gguf");
    if (!check.idempotent) {
        fprintf(stderr, "test-install-idempotency: consistency check not idempotent ops=%d\n",
                check.rebuilt_plan.operation_count);
        for (const auto & issue : check.issues) {
            fprintf(stderr, "  %s\n", issue.c_str());
        }
        return 1;
    }

    printf("test-install-idempotency: OK\n");
    return 0;
}
