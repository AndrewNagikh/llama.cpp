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

static bool validate_plan(
        int n_layers,
        const std::vector<dist_layer_assignment> & a) {
    if (a.empty()) {
        return n_layers == 0;
    }
    if (a.front().layer_start != 0 || a.back().layer_end != n_layers) {
        return false;
    }
    for (size_t i = 1; i < a.size(); ++i) {
        if (a[i].layer_start != a[i - 1].layer_end) {
            return false;
        }
    }
    int seen[1024] = { 0 };
    if (n_layers > 1024) {
        return false;
    }
    for (const auto & as : a) {
        if (as.layer_start < 0 || as.layer_end > n_layers || as.layer_start >= as.layer_end) {
            return false;
        }
        for (int j = as.layer_start; j < as.layer_end; ++j) {
            seen[j]++;
        }
    }
    for (int i = 0; i < n_layers; ++i) {
        if (seen[i] != 1) {
            return false;
        }
    }
    return true;
}

int main() {
    bool ok = true;
    struct test_case {
        int n_layer;
        double total_gb;
        std::vector<dist_planner_node_resources> nodes;
    };

    std::vector<test_case> cases = {
        { 16,  0.8, { make_node("a", 100.0, 2.0, 0.0) } },
        { 80, 40.0, {
            make_node("fast-small", 800.0, 0.0, 10.0),
            make_node("med-med",    300.0, 0.0, 32.0),
            make_node("slow-large", 100.0, 0.0, 40.0),
        }},
        { 32,  4.0, {
            make_node("a", 50.0, 2.0, 0.0),
            make_node("b", 50.0, 2.5, 0.0),
            make_node("c", 50.0, 3.0, 0.0),
        }},
        { 80, 70.0, {
            make_node("gpu-fast",   500.0, 16.0, 24.0),
            make_node("gpu-slow",   200.0, 16.0, 24.0),
            make_node("cpu-buffer",  50.0, 64.0, 0.0),
        }},
    };

    for (size_t ti = 0; ti < cases.size(); ++ti) {
        const auto & tc = cases[ti];
        const auto mem = make_model(tc.n_layer, tc.total_gb);
        const auto result = dist_plan_layers_memory_aware(mem, tc.nodes);
        printf("case %zu: success=%d\n", ti, result.success ? 1 : 0);
        for (const auto & a : result.assignments) {
            printf("  %s %d-%d\n", a.node_id.c_str(), a.layer_start, a.layer_end);
        }
        if (!result.success) {
            fprintf(stderr, "case %zu planning failed: %s\n", ti, result.error.c_str());
            ok = false;
            continue;
        }
        if (!validate_plan(mem.n_layer, result.assignments)) {
            fprintf(stderr, "case %zu plan invalid\n", ti);
            ok = false;
        }
    }

    if (!ok) {
        fprintf(stderr, "test-planner-no-overlap: FAILED\n");
        return 1;
    }
    printf("test-planner-no-overlap: OK\n");
    return 0;
}
