#include "orchestrator/optimizer/cluster_optimizer.h"
#include "test_optimizer_common.h"

#include <cstdio>

int main() {
    const model_manifest manifest = make_test_manifest(16, 30 * 1024 * 1024);

    const std::vector<dist_node_info> baseline = {
        make_opt_node("gpu-a", 100.0, 16.0, 12.0),
    };
    const desired_model_layout current = layout_for_nodes("thr", manifest, baseline);
    if (current.placements.empty()) {
        fprintf(stderr, "test-improvement-threshold: baseline layout failed\n");
        return 1;
    }

    const std::vector<dist_node_info> upgraded = {
        make_opt_node("gpu-a", 100.0, 16.0, 12.0),
        make_opt_node("rtx", 600.0, 32.0, 24.0),
    };

    optimizer_policy policy;
    policy.min_decode_improvement_percent  = 5.0;
    policy.min_prefill_improvement_percent = 5.0;

    const optimization_result result = run_cluster_optimization(
            "thr", manifest, &current, upgraded, policy, 4096);

    if (!result.better || result.decision != optimizer_decision::rebalance) {
        fprintf(stderr, "test-improvement-threshold: expected REBALANCE (decision=%s better=%d)\n",
                optimizer_decision_to_string(result.decision).c_str(), result.better ? 1 : 0);
        return 1;
    }

    printf("test-improvement-threshold: OK decode %.1f -> %.1f\n",
            result.current_estimate.estimated_decode_tps,
            result.candidate_estimate.estimated_decode_tps);
    return 0;
}
