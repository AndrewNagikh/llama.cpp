#include "orchestrator/layout_estimator/layout_estimator.h"

#include <cstdint>
#include <cstdio>

int main() {
    layout_estimate current;
    current.estimated_decode_tps  = 100.0;
    current.estimated_prefill_tps = 200.0;

    layout_estimate candidate = current;
    candidate.estimated_decode_tps  = 104.0;
    candidate.estimated_prefill_tps = 200.0;
    candidate.rebalance_cost_bytes = 1024;

    if (!layout_estimate_is_better(current, candidate, 3.0, 5.0, UINT64_MAX)) {
        fprintf(stderr, "test-optimizer-policy: expected 4%% decode gain to pass 3%% threshold\n");
        return 1;
    }

    candidate.estimated_decode_tps = 103.0;
    if (layout_estimate_is_better(current, candidate, 5.0, 5.0, UINT64_MAX)) {
        fprintf(stderr, "test-optimizer-policy: 3%% gain should not pass 5%% threshold\n");
        return 1;
    }

    candidate.estimated_decode_tps  = 100.0;
    candidate.estimated_prefill_tps = 220.0;
    if (!layout_estimate_is_better(current, candidate, 5.0, 5.0, UINT64_MAX)) {
        fprintf(stderr, "test-optimizer-policy: expected 10%% prefill gain to pass\n");
        return 1;
    }

    candidate.rebalance_cost_bytes = 999999999;
    if (layout_estimate_is_better(current, candidate, 5.0, 5.0, 1000)) {
        fprintf(stderr, "test-optimizer-policy: cost cap should block rebalance\n");
        return 1;
    }

    printf("test-optimizer-policy: OK\n");
    return 0;
}
