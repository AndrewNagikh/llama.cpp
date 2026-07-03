#include "runtime/runtime_role_planner.h"

#include "dist_common.h"

#include <cstdio>
#include <map>

static dist_node_info make_node(
        const std::string & id,
        double score,
        uint64_t ram_gb,
        uint64_t vram_gb,
        bool has_gpu) {
    dist_node_info n{};
    n.node_id = id;
    n.host    = "127.0.0.1";
    n.http_port = 9001;
    n.score   = score;
    n.online  = true;
    n.memory.has_gpu = has_gpu;
    n.memory.free_ram_bytes  = ram_gb * 1024ULL * 1024ULL * 1024ULL;
    n.memory.free_vram_bytes = vram_gb * 1024ULL * 1024ULL * 1024ULL;
    n.caps.has_gpu = has_gpu;
    return n;
}

static int test_role_memory_fit() {
    std::map<std::string, dist_node_info> nodes;
    nodes["mac"] = make_node("mac", 2400, 18, 13, true);
    nodes["win"] = make_node("win", 2200, 31, 11, true);

    model_memory_requirements mem{};
    mem.weights_bytes = 60ULL * 1024ULL * 1024ULL * 1024ULL;
    mem.n_layer       = 48;

    std::vector<dist_layer_assignment> layers;
    dist_layer_assignment a{};
    a.node_id = "mac";
    a.layer_start = 0;
    a.layer_end = 24;
    layers.push_back(a);
    dist_layer_assignment b{};
    b.node_id = "win";
    b.layer_start = 24;
    b.layer_end = 48;
    layers.push_back(b);

    const auto plan = dist_plan_runtime_graph("qwen3-30b", 48, mem, layers, nodes);
    if (!plan.success) {
        fprintf(stderr, "plan failed: %s\n", plan.error.c_str());
        return 1;
    }

    const auto * tok = plan.graph.find_role(runtime_role::tokenizer);
    if (!tok) {
        fprintf(stderr, "missing tokenizer role\n");
        return 1;
    }
    if (tok->node_id == "mac") {
        fprintf(stderr, "tokenizer should not land on 18GB mac for 60GB model\n");
        return 1;
    }
    if (tok->node_id != "win") {
        fprintf(stderr, "expected tokenizer on win, got %s\n", tok->node_id.c_str());
        return 1;
    }
    return 0;
}

static int test_role_assignment() {
    std::map<std::string, dist_node_info> nodes;
    nodes["a"] = make_node("a", 2000, 32, 16, true);
    nodes["b"] = make_node("b", 1800, 32, 16, true);

    model_memory_requirements mem{};
    mem.weights_bytes = 2ULL * 1024ULL * 1024ULL * 1024ULL;
    mem.n_layer       = 16;

    std::vector<dist_layer_assignment> layers;
    dist_layer_assignment la{};
    la.node_id = "a";
    la.layer_start = 0;
    la.layer_end = 8;
    layers.push_back(la);
    la.node_id = "b";
    la.layer_start = 8;
    la.layer_end = 16;
    layers.push_back(la);

    const auto plan = dist_plan_runtime_graph("tinyllama-1.1b", 16, mem, layers, nodes);
    if (!plan.success) {
        fprintf(stderr, "plan failed: %s\n", plan.error.c_str());
        return 1;
    }
    if (plan.graph.pipeline_stages().size() != 2) {
        fprintf(stderr, "expected 2 pipeline stages\n");
        return 1;
    }
    if (!plan.graph.find_role(runtime_role::tokenizer) ||
            !plan.graph.find_role(runtime_role::embedding) ||
            !plan.graph.find_role(runtime_role::output_head) ||
            !plan.graph.find_role(runtime_role::sampler)) {
        fprintf(stderr, "missing service roles\n");
        return 1;
    }
    return 0;
}

int main() {
    if (test_role_memory_fit() != 0) {
        return 1;
    }
    if (test_role_assignment() != 0) {
        return 1;
    }
    fprintf(stderr, "test-runtime-role-planner: OK\n");
    return 0;
}
