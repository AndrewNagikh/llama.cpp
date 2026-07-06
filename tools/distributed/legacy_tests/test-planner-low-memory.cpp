#include "layer_planner.h"
#include "memory_estimator.h"

#include <cstdio>
#include <string>
#include <vector>

static model_memory_requirements make_model(int32_t n_layer, double total_gb) {
    model_memory_requirements mem{};
    mem.n_layer = n_layer;
    const uint64_t total_bytes = static_cast<uint64_t>(total_gb * 1024 * 1024 * 1024);
    mem.weights_bytes = static_cast<uint64_t>(total_bytes * 0.90);
    mem.kv_bytes      = static_cast<uint64_t>(total_bytes * 0.05);
    mem.compute_bytes = static_cast<uint64_t>(total_bytes * 0.03);
    mem.scratch_bytes = static_cast<uint64_t>(total_bytes * 0.02);
    const uint64_t per_layer = total_bytes / static_cast<uint64_t>(n_layer);
    mem.layers.reserve(n_layer);
    for (int32_t i = 0; i < n_layer; ++i) {
        model_layer_memory l{};
        l.layer_index  = i;
        l.weight_bytes = per_layer;
        mem.layers.push_back(l);
    }
    return mem;
}

static dist_planner_node_resources make_node(
        const std::string & id,
        double score,
        double ram_gb,
        double vram_gb) {
    dist_planner_node_resources n{};
    n.node_id = id;
    n.score   = score;
    n.cpu_budget_bytes = static_cast<uint64_t>(ram_gb * 1024 * 1024 * 1024);
    n.gpu_budget_bytes = static_cast<uint64_t>(vram_gb * 1024 * 1024 * 1024);
    n.has_gpu = (vram_gb > 0.0);
    n.backend = n.has_gpu ? "cuda" : "cpu";
    return n;
}

static uint64_t cost_of_range(const model_memory_requirements & mem, int l, int r) {
    uint64_t overhead = mem.kv_bytes + mem.compute_bytes + mem.scratch_bytes;
    uint64_t per_layer_overhead = mem.n_layer > 0
            ? overhead / static_cast<uint64_t>(mem.n_layer) : 0;
    uint64_t sum = 0;
    for (int i = l; i < r; ++i) {
        sum += mem.layers[(size_t) i].weight_bytes + per_layer_overhead;
    }
    return sum;
}

int main() {
    bool ok = true;

    // Model is 20 GB total over 20 layers -> ~1 GB/layer.
    const auto mem = make_model(20, 20.0);

    std::vector<dist_planner_node_resources> nodes = {
        make_node("gpu", 400.0, 0.0, 12.0),
        make_node("cpu", 100.0, 16.0, 0.0),
    };

    const auto result = dist_plan_layers_memory_aware(mem, nodes);
    if (!result.success) {
        fprintf(stderr, "test-planner-low-memory: planning failed: %s\n", result.error.c_str());
        return 1;
    }

    for (const auto & a : result.assignments) {
        const uint64_t used = cost_of_range(mem, a.layer_start, a.layer_end);
        const uint64_t budget = (a.device_hint == "gpu")
                ? nodes[0].gpu_budget_bytes
                : nodes[1].cpu_budget_bytes;
        printf("node %s layers %d-%d device=%s used=%.1f GB budget=%.1f GB\n",
               a.node_id.c_str(), a.layer_start, a.layer_end, a.device_hint.c_str(),
               dist_bytes_to_gb(used), dist_bytes_to_gb(budget));
        if (used > budget) {
            fprintf(stderr, "test-planner-low-memory: node %s exceeds budget\n", a.node_id.c_str());
            ok = false;
        }
    }

    if (!ok) {
        fprintf(stderr, "test-planner-low-memory: FAILED\n");
        return 1;
    }
    printf("test-planner-low-memory: OK\n");
    return 0;
}
