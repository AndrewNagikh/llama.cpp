#include "orchestrator/optimizer/cluster_optimizer.h"
#include "test_optimizer_common.h"

#include <cstdio>

static int layers_on_node(const desired_model_layout & layout, const std::string & node_id) {
    int count = 0;
    for (const auto & p : layout.placements) {
        if (p.node_id == node_id) {
            ++count;
        }
    }
    return count;
}

int main() {
    const model_manifest manifest = make_test_manifest(12, 80 * 1024 * 1024);

    dist_node_info gpu_a = make_opt_node("gpu-a", 250.0, 16.0, 16.0);
    dist_node_info gpu_b = make_opt_node("gpu-b", 240.0, 16.0, 16.0);

    const std::vector<dist_node_info> full = { gpu_a, gpu_b };
    const desired_model_layout current = layout_for_nodes("mem", manifest, full);
    if (current.placements.empty()) {
        fprintf(stderr, "test-memory-change: layout failed\n");
        return 1;
    }

    const int layers_on_a_before = layers_on_node(current, "gpu-a");
    if (layers_on_a_before <= 0) {
        fprintf(stderr, "test-memory-change: expected gpu-a to hold layers initially\n");
        return 1;
    }

    gpu_a.memory.free_vram_bytes = static_cast<uint64_t>(0.2 * 1024.0 * 1024.0 * 1024.0);
    const std::vector<dist_node_info> reduced = { gpu_a, gpu_b };

    optimizer_policy policy;
    policy.min_decode_improvement_percent  = 0.0;
    policy.min_prefill_improvement_percent = 0.0;

    const optimization_result result = run_cluster_optimization(
            "mem", manifest, &current, reduced, policy, 4096);

    if (result.decision != optimizer_decision::rebalance) {
        fprintf(stderr, "test-memory-change: expected REBALANCE after VRAM drop (got %s)\n",
                optimizer_decision_to_string(result.decision).c_str());
        return 1;
    }

    const int layers_on_a_after = layers_on_node(result.candidate_layout, "gpu-a");
    if (layers_on_a_after >= layers_on_a_before) {
        fprintf(stderr, "test-memory-change: expected fewer layers on gpu-a after VRAM drop (%d -> %d)\n",
                layers_on_a_before, layers_on_a_after);
        return 1;
    }

    printf("test-memory-change: OK gpu-a layers %d -> %d decision=%s\n",
            layers_on_a_before, layers_on_a_after,
            optimizer_decision_to_string(result.decision).c_str());
    return 0;
}
