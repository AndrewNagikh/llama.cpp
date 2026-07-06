#include "orchestrator/coverage/coverage.h"
#include "test_coverage_common.h"

#include <cstdio>

int main() {
    const desired_model_layout desired = make_desired_layout("m", {
        { "a", 0 }, { "a", 1 }, { "b", 2 },
    });
    const actual_model_layout actual = make_actual_layout("m", {});

    const coverage_report report = compute_coverage(desired, actual);
    if (report.state != coverage_state::empty) {
        fprintf(stderr, "test-empty-cluster: expected EMPTY got %s\n",
                coverage_state_to_string(report.state).c_str());
        return 1;
    }

    printf("test-empty-cluster: OK\n");
    return 0;
}
