#include "layer_planner.h"
#include "memory_estimator.h"

#include <cstdio>
#include <string>
#include <vector>

static model_memory_requirements make_model(
        int32_t n_layer,
        double  total_gb) {
    model_memory_requirements mem{};
    mem.n_layer = n_layer;
    // Spread the total evenly, including a tiny KV/compute share so the model
    // looks like a real transformer to the planner.
    const uint64_t total_bytes = static_cast<uint64_t>(total_gb * 1024 * 1024 * 1024);
    mem.weights_bytes = static_cast<uint64_t>(total_bytes * 0.90);
    mem.kv_bytes      = static_cast<uint64_t>(total_bytes * 0.05);
    mem.compute_bytes = static_cast<uint64_t>(total_bytes * 0.03);
    mem.scratch_bytes = static_cast<uint64_t>(total_bytes * 0.02);
    const uint64_t per_layer = total_bytes / static_cast<uint64_t>(n_layer);
    mem.layers.reserve(n_layer);
    for (int32_t i = 0; i < n_layer; ++i) {
        model_layer_memory l{};
        l.layer_index = i;
        l.weight_bytes = per_layer;
        mem.layers.push_back(l);
    }
    return mem;
}

static dist_planner_node_resources make_node(
        const std::string & id,
        double score,
        double ram_gb,
        double vram_gb,
        const std::string & backend = "cpu") {
    dist_planner_node_resources n{};
    n.node_id = id;
    n.score   = score;
    n.backend = backend;
    n.cpu_budget_bytes = static_cast<uint64_t>(ram_gb * 1024 * 1024 * 1024);
    n.gpu_budget_bytes = static_cast<uint64_t>(vram_gb * 1024 * 1024 * 1024);
    n.has_gpu = (vram_gb > 0.0);
    return n;
}

static bool check_contiguous_no_gaps(int n_layers, const std::vector<dist_layer_assignment> & a) {
    if (a.empty()) {
        return false;
    }
    if (a.front().layer_start != 0) {
        return false;
    }
    if (a.back().layer_end != n_layers) {
        return false;
    }
    for (size_t i = 1; i < a.size(); ++i) {
        if (a[i].layer_start != a[i - 1].layer_end) {
            return false;
        }
    }
    return true;
}

int main() {
    bool ok = true;

    const auto model = make_model(50, 40.0);
    printf("test-memory-aware-planner: required=%.1f GB per_layer=%.2f GB\n",
           model.total_gb(), model.total_gb() / model.n_layer);

    std::vector<dist_planner_node_resources> nodes = {
        make_node("A", 100.0, 16.0, 0.0, "cpu"),
        make_node("B", 300.0, 32.0, 0.0, "cpu"),
        make_node("C", 800.0, 10.0, 0.0, "cpu"),
    };

    const auto result = dist_plan_layers_memory_aware(model, nodes);
    if (!result.success) {
        fprintf(stderr, "test-memory-aware-planner: planning failed: %s\n", result.error.c_str());
        return 1;
    }

    printf("layout:\n");
    for (const auto & a : result.assignments) {
        printf("  %s %d-%d\n", a.node_id.c_str(), a.layer_start, a.layer_end);
    }

    if (!check_contiguous_no_gaps(model.n_layer, result.assignments)) {
        fprintf(stderr, "test-memory-aware-planner: layout is not contiguous\n");
        ok = false;
    }

    // Node C has a 10 GB budget with a ~0.8 GB/layer model, but budget also
    // includes KV/compute/scratch, so its practical layer budget is <= 12.
    const auto * c = (const dist_layer_assignment *) nullptr;
    for (const auto & a : result.assignments) {
        if (a.node_id == "C") {
            c = &a;
            break;
        }
    }
    if (!c) {
        fprintf(stderr, "test-memory-aware-planner: node C not assigned\n");
        ok = false;
    } else if (c->layer_end - c->layer_start > 13) {
        fprintf(stderr, "test-memory-aware-planner: node C assigned too many layers\n");
        ok = false;
    }

    if (!ok) {
        fprintf(stderr, "test-memory-aware-planner: FAILED\n");
        return 1;
    }
    printf("test-memory-aware-planner: OK\n");
    return 0;
}
