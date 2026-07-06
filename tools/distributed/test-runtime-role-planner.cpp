#include "runtime/runtime_role_planner.h"

#include "dist_common.h"
#include "runtime/runtime_descriptor.h"

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
    n.cpu.logical_cores = has_gpu ? 8 : 4;
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

    const runtime_descriptor desc =
            make_basic_text_generation_runtime_descriptor("qwen3-30b", "qwen.arch", 48);
    const auto plan = dist_plan_runtime_graph(desc, mem, layers, nodes);
    if (!plan.success) {
        fprintf(stderr, "plan failed: %s\n", plan.error.c_str());
        return 1;
    }

    const auto * tok = plan.graph.find_role(runtime_role::tokenizer);
    if (!tok) {
        fprintf(stderr, "missing tokenizer role\n");
        return 1;
    }
    if (tok->node_id != "mac") {
        fprintf(stderr, "tokenizer expected on entry boundary mac, got %s\n", tok->node_id.c_str());
        return 1;
    }
    const runtime_execution_graph_validation graph_validation =
            validate_runtime_graph_against_descriptor(desc, plan.graph);
    if (!graph_validation.ok()) {
        fprintf(stderr, "planner graph validation failed: %s\n",
                graph_validation.errors.empty() ? "unknown" : graph_validation.errors.front().c_str());
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

    const runtime_descriptor desc =
            make_basic_text_generation_runtime_descriptor("tinyllama-1.1b", "llama.arch", 16);
    const auto plan = dist_plan_runtime_graph(desc, mem, layers, nodes);
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
    const runtime_execution_graph_validation graph_validation =
            validate_runtime_graph_against_descriptor(desc, plan.graph);
    if (!graph_validation.ok()) {
        fprintf(stderr, "assignment graph validation failed: %s\n",
                graph_validation.errors.empty() ? "unknown" : graph_validation.errors.front().c_str());
        return 1;
    }
    return 0;
}

static int test_large_model_service_offload() {
    std::map<std::string, dist_node_info> nodes;
    nodes["entry"] = make_node("entry", 3000, 16, 12, true);
    nodes["mid"]   = make_node("mid", 2500, 32, 16, true);
    nodes["final"] = make_node("final", 2400, 32, 16, true);

    model_memory_requirements mem{};
    mem.weights_bytes = 30ULL * 1024ULL * 1024ULL * 1024ULL;
    mem.n_layer       = 32;

    std::vector<dist_layer_assignment> layers;
    dist_layer_assignment e{};
    e.node_id = "entry";
    e.layer_start = 0;
    e.layer_end = 10;
    layers.push_back(e);
    dist_layer_assignment m{};
    m.node_id = "mid";
    m.layer_start = 10;
    m.layer_end = 22;
    layers.push_back(m);
    dist_layer_assignment f{};
    f.node_id = "final";
    f.layer_start = 22;
    f.layer_end = 32;
    layers.push_back(f);

    const runtime_descriptor desc =
            make_basic_text_generation_runtime_descriptor("qwen-30b", "qwen.arch", 32);
    const auto plan = dist_plan_runtime_graph(desc, mem, layers, nodes);
    if (!plan.success) {
        fprintf(stderr, "large model plan failed: %s\n", plan.error.c_str());
        return 1;
    }

    const auto * tok = plan.graph.find_role(runtime_role::tokenizer);
    const auto * emb = plan.graph.find_role(runtime_role::embedding);
    const auto * out = plan.graph.find_role(runtime_role::output_head);
    if (!tok || !emb || !out) {
        fprintf(stderr, "missing service roles in large model plan\n");
        return 1;
    }
    if (tok->node_id != "entry") {
        fprintf(stderr, "tokenizer expected on entry boundary, got %s\n", tok->node_id.c_str());
        return 1;
    }
    if (emb->node_id != "entry") {
        fprintf(stderr, "embedding expected on entry boundary, got %s\n", emb->node_id.c_str());
        return 1;
    }
    if (out->node_id != "final") {
        fprintf(stderr, "output_head expected on final boundary, got %s\n", out->node_id.c_str());
        return 1;
    }
    const runtime_execution_graph_validation graph_validation =
            validate_runtime_graph_against_descriptor(desc, plan.graph);
    if (!graph_validation.ok()) {
        fprintf(stderr, "large model graph validation failed: %s\n",
                graph_validation.errors.empty() ? "unknown" : graph_validation.errors.front().c_str());
        return 1;
    }
    return 0;
}

static int test_legacy_wrapper_descriptor_equivalence() {
    std::map<std::string, dist_node_info> nodes;
    nodes["solo"] = make_node("solo", 1200, 16, 0, false);

    model_memory_requirements mem{};
    mem.weights_bytes = 1ULL * 1024ULL * 1024ULL * 1024ULL;
    mem.n_layer       = 4;

    std::vector<dist_layer_assignment> layers;
    dist_layer_assignment la{};
    la.node_id = "solo";
    la.layer_start = 0;
    la.layer_end = 4;
    layers.push_back(la);

    const runtime_descriptor desc =
            make_basic_text_generation_runtime_descriptor("single-node", "llama.arch", 4);
    const auto descriptor_plan = dist_plan_runtime_graph(desc, mem, layers, nodes);
    const auto wrapper_plan = dist_plan_runtime_graph("single-node", 4, mem, layers, nodes);
    if (!descriptor_plan.success || !wrapper_plan.success) {
        fprintf(stderr, "equivalence planning failed\n");
        return 1;
    }
    if (descriptor_plan.graph.assignments.size() != wrapper_plan.graph.assignments.size()) {
        fprintf(stderr, "equivalence assignment count mismatch\n");
        return 1;
    }
    for (size_t i = 0; i < descriptor_plan.graph.assignments.size(); ++i) {
        const runtime_role_assignment & a = descriptor_plan.graph.assignments[i];
        const runtime_role_assignment & b = wrapper_plan.graph.assignments[i];
        if (a.role != b.role || a.node_id != b.node_id ||
                a.layer_start != b.layer_start ||
                a.layer_end != b.layer_end) {
            fprintf(stderr, "equivalence mismatch at assignment %zu\n", i);
            return 1;
        }
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
    if (test_large_model_service_offload() != 0) {
        return 1;
    }
    if (test_legacy_wrapper_descriptor_equivalence() != 0) {
        return 1;
    }
    fprintf(stderr, "test-runtime-role-planner: OK\n");
    return 0;
}
