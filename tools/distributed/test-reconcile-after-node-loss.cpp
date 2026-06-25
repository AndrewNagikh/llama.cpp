#include "orchestrator/coverage/coverage.h"
#include "orchestrator/model_registry.h"
#include "test_coverage_common.h"

#include <cstdio>
#include <set>

int main() {
    cluster_model_registry reg;
    dist_model_record record;
    record.model_id = "m";
    reg.add_or_update(record);

    const desired_model_layout desired = make_desired_layout("m", {
        { "a", 0 }, { "b", 1 }, { "b", 2 },
    });
    reg.apply_layout("m", desired, nullptr);

    const actual_model_layout before = make_actual_layout("m", {
        make_ready_layer(0, "a"),
        make_ready_layer(1, "b"),
        make_ready_layer(2, "b"),
    });
    reg.apply_actual("m", before, nullptr);
    reg.refresh_coverage("m", { "a", "b" }, nullptr);
    if (reg.find("m")->coverage->state != coverage_state::ready) {
        fprintf(stderr, "test-reconcile-after-node-loss: expected initial READY\n");
        return 1;
    }

    const std::set<std::string> online_after_loss = { "a" };
    const reconciliation_result result = reconcile_layers(
            desired, before, online_after_loss);
    if (result.state != coverage_state::degraded) {
        fprintf(stderr, "test-reconcile-after-node-loss: expected DEGRADED\n");
        return 1;
    }
    if (result.missing.empty()) {
        fprintf(stderr, "test-reconcile-after-node-loss: expected missing layers\n");
        return 1;
    }

    reg.refresh_coverage("m", online_after_loss, nullptr);
    if (reg.find("m")->coverage->state != coverage_state::degraded) {
        fprintf(stderr, "test-reconcile-after-node-loss: registry coverage not DEGRADED\n");
        return 1;
    }

    printf("test-reconcile-after-node-loss: OK missing=%zu\n", result.missing.size());
    return 0;
}
