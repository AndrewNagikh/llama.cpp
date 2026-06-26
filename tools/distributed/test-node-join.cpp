#include "orchestrator/optimizer/cluster_optimizer.h"
#include "test_optimizer_common.h"

#include <cstdio>

int main() {
    const model_manifest manifest = make_test_manifest(12, 35 * 1024 * 1024);

    const std::vector<dist_node_info> one = {
        make_opt_node("gpu-a", 200.0, 16.0, 16.0),
    };
    const desired_model_layout current = layout_for_nodes("join", manifest, one);
    if (current.placements.empty()) {
        fprintf(stderr, "test-node-join: initial layout failed\n");
        return 1;
    }

    const std::vector<dist_node_info> joined = {
        make_opt_node("gpu-a", 200.0, 16.0, 16.0),
        make_opt_node("rtx-5090", 800.0, 64.0, 32.0),
    };

    const optimization_result result = run_cluster_optimization(
            "join", manifest, &current, joined, {}, 4096);

    if (result.decision != optimizer_decision::rebalance || !result.better) {
        fprintf(stderr, "test-node-join: expected REBALANCE after powerful GPU join\n");
        return 1;
    }
    if (result.candidate_layout.placements.empty()) {
        fprintf(stderr, "test-node-join: missing candidate layout\n");
        return 1;
    }

    printf("test-node-join: OK candidate_decode=%.1f\n",
            result.candidate_estimate.estimated_decode_tps);
    return 0;
}
