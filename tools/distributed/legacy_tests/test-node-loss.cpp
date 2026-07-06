#include "orchestrator/optimizer/cluster_optimizer.h"
#include "test_optimizer_common.h"

#include <cstdio>

int main() {
    const model_manifest manifest = make_test_manifest(10, 30 * 1024 * 1024);

    const std::vector<dist_node_info> pair = {
        make_opt_node("gpu-a", 300.0, 16.0, 16.0),
        make_opt_node("gpu-b", 280.0, 16.0, 16.0),
    };
    const desired_model_layout current = layout_for_nodes("loss", manifest, pair);
    if (current.placements.empty()) {
        fprintf(stderr, "test-node-loss: initial layout failed\n");
        return 1;
    }

    const std::vector<dist_node_info> after_loss = {
        make_opt_node("gpu-a", 300.0, 16.0, 16.0),
        make_opt_node("gpu-b", 280.0, 16.0, 16.0, "cuda", false),
    };

    const optimization_result result = run_cluster_optimization(
            "loss", manifest, &current, after_loss, {}, 4096);

    if (result.decision != optimizer_decision::rebalance) {
        fprintf(stderr, "test-node-loss: expected REBALANCE after node loss (got %s)\n",
                optimizer_decision_to_string(result.decision).c_str());
        return 1;
    }

    printf("test-node-loss: OK\n");
    return 0;
}
