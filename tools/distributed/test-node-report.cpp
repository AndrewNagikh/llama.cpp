#include "orchestrator/coverage/coverage.h"
#include "orchestrator/model_registry.h"
#include "test_coverage_common.h"

#include <cstdio>

int main() {
    cluster_model_registry reg;
    dist_model_record record;
    record.model_id = "m";
    reg.add_or_update(record);

    const desired_model_layout desired = make_desired_layout("m", {
        { "node-a", 0 }, { "node-a", 1 },
    });
    reg.apply_layout("m", desired, nullptr);

    const actual_model_layout node_report = make_actual_layout("m", {
        make_ready_layer(0, "node-a"),
        make_ready_layer(1, "node-a"),
    });
    if (!reg.apply_actual("m", node_report, nullptr)) {
        fprintf(stderr, "test-node-report: apply_actual failed\n");
        return 1;
    }

    const auto * stored = reg.find("m");
    if (!stored || !stored->actual.has_value() || stored->actual->layers.size() != 2) {
        fprintf(stderr, "test-node-report: actual not stored\n");
        return 1;
    }

    printf("test-node-report: OK layers=%zu\n", stored->actual->layers.size());
    return 0;
}
