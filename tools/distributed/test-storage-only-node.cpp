#include "orchestrator/optimizer/cluster_optimizer.h"
#include "test_optimizer_common.h"

#include <cstdio>

int main() {
    const std::vector<dist_node_info> nodes = {
        make_opt_node("fast-gpu", 500.0, 16.0, 24.0),
        make_opt_node("slow-cpu", 8.0, 32.0, 0.0, "cpu"),
    };

    std::vector<optimizer_node> opt_nodes;
    for (const auto & n : nodes) {
        opt_nodes.push_back(optimizer_node_from_dist(n));
    }

    optimizer_policy policy;
    policy.allow_storage_only_nodes = true;
    policy.storage_only_score_ratio = 0.25;

    const double median = 500.0;
    const auto roles = classify_node_roles(opt_nodes, policy, median);

    const auto slow_it = roles.find("slow-cpu");
    if (slow_it == roles.end() || slow_it->second != node_role::storage_only) {
        fprintf(stderr, "test-storage-only-node: expected slow-cpu STORAGE_ONLY\n");
        return 1;
    }

    const auto fast_it = roles.find("fast-gpu");
    if (fast_it == roles.end() || fast_it->second != node_role::active) {
        fprintf(stderr, "test-storage-only-node: expected fast-gpu ACTIVE\n");
        return 1;
    }

    printf("test-storage-only-node: OK\n");
    return 0;
}
