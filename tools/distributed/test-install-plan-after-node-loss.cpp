#include "orchestrator/install_planner/install_planner.h"
#include "test_install_common.h"

#include <cstdio>
#include <set>

int main() {
    const model_manifest manifest = make_manifest_with_layer_ranges(6, 20 * 1024 * 1024);
    const desired_model_layout desired = make_desired_layout("m", {
        { "a", 0 }, { "a", 1 }, { "lost", 2 }, { "lost", 3 },
    });
    const actual_model_layout actual = make_actual_layout("m", {
        make_ready_layer(0, "a"),
        make_ready_layer(1, "a"),
    });
    const coverage_report coverage = make_coverage("m", { 2, 3 }, {}, 4);

    const std::set<std::string> online = { "a" };
    const auto result = build_install_plan(manifest, desired, actual, coverage);
    if (!result.success) {
        fprintf(stderr, "test-install-plan-after-node-loss: %s\n", result.error.c_str());
        return 1;
    }

    int download_for_missing = 0;
    for (const auto & op : result.plan.operations) {
        if (op.action == install_action::download &&
                (op.layer_index == 2 || op.layer_index == 3) &&
                op.node_id == "lost") {
            ++download_for_missing;
        }
    }
    if (download_for_missing != 2) {
        fprintf(stderr, "test-install-plan-after-node-loss: expected 2 downloads for lost node\n");
        return 1;
    }

    printf("test-install-plan-after-node-loss: OK downloads=%d\n", download_for_missing);
    return 0;
}
