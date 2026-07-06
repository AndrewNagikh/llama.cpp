#include "orchestrator/optimizer/cluster_optimizer.h"
#include "test_optimizer_common.h"

#include <cstdio>

int main() {
    const model_manifest manifest = make_test_manifest(12, 40 * 1024 * 1024);

    const std::vector<dist_node_info> single = {
        make_opt_node("gpu-a", 120.0, 16.0, 16.0),
    };
    const desired_model_layout current = layout_for_nodes("sim", manifest, single);
    if (current.placements.empty()) {
        fprintf(stderr, "test-performance-simulation: initial layout failed\n");
        return 1;
    }

    const std::vector<dist_node_info> cluster = {
        make_opt_node("gpu-a", 120.0, 16.0, 16.0),
        make_opt_node("gpu-b", 480.0, 16.0, 24.0),
    };

    const optimization_result result = run_cluster_optimization(
            "sim", manifest, &current, cluster, {}, 4096);

    if (result.current_estimate.estimated_decode_tps <= 0.0) {
        fprintf(stderr, "test-performance-simulation: missing current estimate\n");
        return 1;
    }
    if (result.candidate_estimate.estimated_decode_tps <= 0.0) {
        fprintf(stderr, "test-performance-simulation: missing candidate estimate\n");
        return 1;
    }
    if (result.decision != optimizer_decision::rebalance || !result.better) {
        fprintf(stderr, "test-performance-simulation: expected REBALANCE with benefit\n");
        return 1;
    }

    printf("test-performance-simulation: OK current=%.1f candidate=%.1f decision=%s\n",
            result.current_estimate.estimated_decode_tps,
            result.candidate_estimate.estimated_decode_tps,
            optimizer_decision_to_string(result.decision).c_str());
    return 0;
}
