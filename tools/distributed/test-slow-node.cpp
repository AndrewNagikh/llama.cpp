#include "orchestrator/optimizer/cluster_optimizer.h"
#include "test_optimizer_common.h"

#include <cstdio>

int main() {
    const model_manifest manifest = make_test_manifest(8, 25 * 1024 * 1024);

    const std::vector<dist_node_info> cluster = {
        make_opt_node("gpu-main", 350.0, 16.0, 16.0),
        make_opt_node("intel-n100", 4.0, 32.0, 0.0, "cpu"),
    };

    const desired_model_layout current = layout_for_nodes("slow", manifest, { cluster[0] });
    if (current.placements.empty()) {
        fprintf(stderr, "test-slow-node: layout failed\n");
        return 1;
    }

    const optimization_result result = run_cluster_optimization(
            "slow", manifest, &current, cluster, {}, 4096);

    const auto role_it = result.node_roles.find("intel-n100");
    if (role_it == result.node_roles.end() ||
            role_it->second != node_role::storage_only) {
        fprintf(stderr, "test-slow-node: expected intel-n100 STORAGE_ONLY\n");
        return 1;
    }

    if (result.decision == optimizer_decision::rebalance) {
        fprintf(stderr, "test-slow-node: inference layout should stay unchanged\n");
        return 1;
    }

    printf("test-slow-node: OK decision=%s\n",
            optimizer_decision_to_string(result.decision).c_str());
    return 0;
}
