#include "orchestrator/coverage/coverage.h"
#include "test_coverage_common.h"

#include <cstdio>
#include <set>

int main() {
    const desired_model_layout desired = make_desired_layout("m", {
        { "a", 0 }, { "a", 1 }, { "lost", 2 }, { "lost", 3 },
    });
    const actual_model_layout actual = make_actual_layout("m", {
        make_ready_layer(0, "a"),
        make_ready_layer(1, "a"),
    });

    const std::set<std::string> online = { "a" };
    const coverage_report report = compute_coverage(desired, actual, online);
    if (report.state != coverage_state::degraded) {
        fprintf(stderr, "test-degraded-coverage: expected DEGRADED got %s\n",
                coverage_state_to_string(report.state).c_str());
        return 1;
    }
    if (report.missing_layers < 2) {
        fprintf(stderr, "test-degraded-coverage: expected missing layers on lost node\n");
        return 1;
    }

    printf("test-degraded-coverage: OK missing=%d\n", report.missing_layers);
    return 0;
}
