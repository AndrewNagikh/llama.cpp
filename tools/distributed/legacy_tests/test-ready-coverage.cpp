#include "orchestrator/coverage/coverage.h"
#include "test_coverage_common.h"

#include <cstdio>

int main() {
    const desired_model_layout desired = make_desired_layout("m", {
        { "a", 0 }, { "a", 1 }, { "b", 2 },
    });
    const actual_model_layout actual = make_actual_layout("m", {
        make_ready_layer(0, "a"),
        make_ready_layer(1, "a"),
        make_ready_layer(2, "b"),
    });

    const coverage_report report = compute_coverage(desired, actual);
    if (report.state != coverage_state::ready) {
        fprintf(stderr, "test-ready-coverage: expected READY got %s\n",
                coverage_state_to_string(report.state).c_str());
        return 1;
    }

    std::string err;
    if (!coverage_has_no_extra_layers(desired, actual, err)) {
        fprintf(stderr, "test-ready-coverage: %s\n", err.c_str());
        return 1;
    }

    printf("test-ready-coverage: OK\n");
    return 0;
}
