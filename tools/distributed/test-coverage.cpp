#include "orchestrator/coverage/coverage.h"
#include "test_coverage_common.h"

#include <cstdio>

int main() {
    const desired_model_layout desired = make_desired_layout("m", {
        { "a", 0 }, { "a", 1 }, { "b", 2 }, { "b", 3 },
    });
    const actual_model_layout actual = make_actual_layout("m", {
        make_ready_layer(0, "a"),
        make_ready_layer(1, "a"),
        make_ready_layer(2, "b"),
    });

    const coverage_report report = compute_coverage(desired, actual);
    if (report.state != coverage_state::partial) {
        fprintf(stderr, "test-coverage: expected PARTIAL got %s\n",
                coverage_state_to_string(report.state).c_str());
        return 1;
    }
    if (report.ready_layers != 3 || report.missing_layers != 1) {
        fprintf(stderr, "test-coverage: unexpected counts ready=%d missing=%d\n",
                report.ready_layers, report.missing_layers);
        return 1;
    }

    printf("test-coverage: OK state=%s\n", coverage_state_to_string(report.state).c_str());
    return 0;
}
