#include "orchestrator/optimizer/cluster_optimizer.h"
#include "test_optimizer_common.h"

#include <cstdio>

int main() {
    const model_manifest manifest = make_test_manifest(8, 20 * 1024 * 1024);

    const std::vector<dist_node_info> active = {
        make_opt_node("gpu-a", 400.0, 16.0, 16.0),
        make_opt_node("gpu-b", 380.0, 16.0, 16.0),
    };
    const desired_model_layout current = layout_for_nodes("nb", manifest, active);
    if (current.placements.empty()) {
        fprintf(stderr, "test-no-benefit: layout build failed\n");
        return 1;
    }

    const std::vector<dist_node_info> with_slow = {
        make_opt_node("gpu-a", 400.0, 16.0, 16.0),
        make_opt_node("gpu-b", 380.0, 16.0, 16.0),
        make_opt_node("n100", 6.0, 32.0, 0.0, "cpu"),
    };

    optimizer_policy policy;
    policy.min_decode_improvement_percent  = 5.0;
    policy.min_prefill_improvement_percent = 5.0;

    const optimization_result result = run_cluster_optimization(
            "nb", manifest, &current, with_slow, policy, 4096);

    if (result.decision == optimizer_decision::rebalance) {
        fprintf(stderr, "test-no-benefit: weak node should not force rebalance\n");
        return 1;
    }

    const auto role_it = result.node_roles.find("n100");
    if (role_it == result.node_roles.end() ||
            role_it->second != node_role::storage_only) {
        fprintf(stderr, "test-no-benefit: expected n100 STORAGE_ONLY\n");
        return 1;
    }

    printf("test-no-benefit: OK decision=%s\n",
            optimizer_decision_to_string(result.decision).c_str());
    return 0;
}
