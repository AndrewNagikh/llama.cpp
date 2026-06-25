#include "orchestrator/coverage/coverage.h"
#include "test_coverage_common.h"

#include <cstdio>

int main() {
    const desired_model_layout desired = make_desired_layout("m", {
        { "a", 0 }, { "a", 1 }, { "b", 2 },
    });
    const actual_model_layout actual = make_actual_layout("m", {
        make_ready_layer(0, "a"),
    });

    const reconciliation_result result = reconcile_layers(desired, actual);
    if (result.missing.size() != 2) {
        fprintf(stderr, "test-reconcile: expected 2 missing layers\n");
        return 1;
    }
    if (result.missing[0] != 1 || result.missing[1] != 2) {
        fprintf(stderr, "test-reconcile: unexpected missing list\n");
        return 1;
    }

    printf("test-reconcile: OK missing=%zu\n", result.missing.size());
    return 0;
}
