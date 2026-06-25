#include "orchestrator/coverage/coverage.h"
#include "test_coverage_common.h"

#include <cstdio>

int main() {
    const desired_model_layout desired = make_desired_layout("m", {
        { "a", 0 }, { "a", 1 },
    });

    installed_layer bad = make_ready_layer(0, "a", "checksum-ok");
    bad.state = install_state::corrupted;
    const actual_model_layout actual = make_actual_layout("m", { bad, make_ready_layer(1, "a") });

    const coverage_report report = compute_coverage(desired, actual);
    if (report.state != coverage_state::degraded) {
        fprintf(stderr, "test-checksum: expected DEGRADED got %s\n",
                coverage_state_to_string(report.state).c_str());
        return 1;
    }
    if (report.corrupted_layers != 1 || report.corrupted[0] != 0) {
        fprintf(stderr, "test-checksum: corrupted layer not detected\n");
        return 1;
    }

    printf("test-checksum: OK\n");
    return 0;
}
