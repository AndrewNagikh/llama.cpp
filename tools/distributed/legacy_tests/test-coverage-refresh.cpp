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
        { "a", 0 }, { "a", 1 }, { "b", 2 },
    });
    reg.apply_layout("m", desired, nullptr);

    const actual_model_layout partial = make_actual_layout("m", {
        make_ready_layer(0, "a"),
    });
    reg.apply_actual("m", partial, nullptr);
    if (!reg.refresh_coverage("m", { "a", "b" }, nullptr)) {
        fprintf(stderr, "test-coverage-refresh: refresh failed\n");
        return 1;
    }

    const auto * stored = reg.find("m");
    if (!stored || !stored->coverage.has_value()) {
        fprintf(stderr, "test-coverage-refresh: coverage missing\n");
        return 1;
    }
    if (stored->coverage->state != coverage_state::partial) {
        fprintf(stderr, "test-coverage-refresh: expected PARTIAL\n");
        return 1;
    }

    const actual_model_layout full = make_actual_layout("m", {
        make_ready_layer(0, "a"),
        make_ready_layer(1, "a"),
        make_ready_layer(2, "b"),
    });
    reg.apply_actual("m", full, nullptr);
    reg.refresh_coverage("m", { "a", "b" }, nullptr);

    if (reg.find("m")->coverage->state != coverage_state::ready) {
        fprintf(stderr, "test-coverage-refresh: expected READY after update\n");
        return 1;
    }

    printf("test-coverage-refresh: OK\n");
    return 0;
}
