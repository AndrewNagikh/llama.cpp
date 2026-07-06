#include "orchestrator/consistency/cluster_consistency.h"
#include "orchestrator/coverage/coverage.h"
#include "test_coverage_common.h"

#include <cstdio>

int main() {
    const desired_model_layout desired = make_desired_layout("cov-m", {
        { "a", 0 }, { "a", 1 }, { "b", 2 },
    });
    const actual_model_layout actual = make_actual_layout("cov-m", {
        make_ready_layer(0, "a"),
        make_ready_layer(1, "a"),
        make_ready_layer(2, "b"),
    });

    const coverage_report before = compute_coverage(desired, actual);
    if (before.state != coverage_state::ready) {
        fprintf(stderr, "test-coverage-consistency: expected READY baseline\n");
        return 1;
    }

    coverage_report after = before;
    std::string err;
    if (!assert_coverage_stable(before, after, true, err)) {
        fprintf(stderr, "test-coverage-consistency: stable identical coverage rejected: %s\n",
                err.c_str());
        return 1;
    }

    after = before;
    after.ready_layers   = 1;
    after.missing_layers = 2;
    after.missing        = { 1, 2 };
    after.state          = coverage_state::partial;
    if (assert_coverage_stable(before, after, true, err)) {
        fprintf(stderr, "test-coverage-consistency: expected regression detection\n");
        return 1;
    }

    after = before;
    after.ready_layers = 2;
    after.state        = coverage_state::partial;
    if (assert_coverage_stable(before, after, true, err)) {
        fprintf(stderr, "test-coverage-consistency: expected ready_layers decrease detection\n");
        return 1;
    }

    after.ready_layers = 4;
    after.state        = coverage_state::ready;
    if (!assert_coverage_stable(before, after, false, err)) {
        fprintf(stderr, "test-coverage-consistency: store change should allow coverage update\n");
        return 1;
    }

    printf("test-coverage-consistency: OK\n");
    return 0;
}
