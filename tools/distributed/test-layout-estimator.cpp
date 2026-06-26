#include "orchestrator/layout_estimator/layout_estimator.h"
#include "test_layout_common.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <map>

int main() {
    const model_manifest manifest = make_test_manifest(8, 32 * 1024 * 1024);

    const std::vector<layout_node_input> nodes = {
        make_layout_node("gpu-a", 400.0, 8.0, 12.0, "cuda"),
        make_layout_node("gpu-b", 200.0, 8.0, 8.0, "cuda"),
    };
    const auto built = build_desired_layout("m", manifest, nodes);
    if (!built.success) {
        fprintf(stderr, "test-layout-estimator: layout build failed\n");
        return 1;
    }

    std::map<std::string, double> decode = { { "gpu-a", 400.0 }, { "gpu-b", 200.0 } };
    std::map<std::string, double> prefill = { { "gpu-a", 600.0 }, { "gpu-b", 300.0 } };

    const layout_estimate est = estimate_layout_performance(built.layout, decode, prefill);
    if (est.estimated_decode_tps < 199.0 || est.estimated_decode_tps > 201.0) {
        fprintf(stderr, "test-layout-estimator: expected decode bottleneck ~200, got %.1f\n",
                est.estimated_decode_tps);
        return 1;
    }

    desired_model_layout shifted = built.layout;
    for (auto & p : shifted.placements) {
        if (p.layer_index % 2 == 0) {
            p.node_id = "gpu-b";
        } else {
            p.node_id = "gpu-a";
        }
    }

    const uint64_t cost = estimate_rebalance_cost_bytes(built.layout, shifted, manifest);
    if (cost == 0) {
        fprintf(stderr, "test-layout-estimator: expected non-zero rebalance cost\n");
        return 1;
    }

    layout_estimate current = est;
    layout_estimate candidate = current;
    candidate.estimated_decode_tps  = current.estimated_decode_tps * 1.08;
    candidate.estimated_prefill_tps = current.estimated_prefill_tps * 1.02;
    candidate.rebalance_cost_bytes  = cost;

    if (!layout_estimate_is_better(current, candidate, 5.0, 5.0, UINT64_MAX)) {
        fprintf(stderr, "test-layout-estimator: expected 8%% decode gain to pass 5%% threshold\n");
        return 1;
    }

    printf("test-layout-estimator: OK decode=%.1f cost=%llu\n",
            est.estimated_decode_tps, static_cast<unsigned long long>(cost));
    return 0;
}
