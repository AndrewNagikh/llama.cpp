#include "runtime/runtime_role_planner.h"

#include "dist_common.h"

#include <cstdio>
#include <map>

static dist_node_info make_node(const std::string & id, double score) {
    dist_node_info n{};
    n.node_id   = id;
    n.host      = "127.0.0.1";
    n.http_port = 9001;
    n.score     = score;
    n.online    = true;
    n.memory.free_ram_bytes = 32ULL * 1024ULL * 1024ULL * 1024ULL;
    n.cpu.logical_cores = 8;
    return n;
}

static int test_services_colocated_with_entry_boundary() {
    std::vector<dist_layer_assignment> layers;
    dist_layer_assignment a{};
    a.node_id     = "node-a";
    a.layer_start = 0;
    a.layer_end   = 8;
    a.score       = 100.0;
    layers.push_back(a);

    dist_layer_assignment b{};
    b.node_id     = "node-b";
    b.layer_start = 8;
    b.layer_end   = 16;
    b.score       = 50.0;
    layers.push_back(b);

    std::map<std::string, dist_node_info> nodes;
    nodes["node-a"] = make_node("node-a", 100.0);
    nodes["node-b"] = make_node("node-b", 50.0);

    model_memory_requirements mem{};
    mem.weights_bytes = 8ull * 1024 * 1024 * 1024;

    const auto plan = dist_plan_runtime_graph("test-model", 16, mem, layers, nodes);
    if (!plan.success) {
        fprintf(stderr, "plan failed: %s\n", plan.error.c_str());
        return 1;
    }

    const auto * emb = plan.graph.find_role(runtime_role::embedding);
    const auto * tok = plan.graph.find_role(runtime_role::tokenizer);
    if (emb == nullptr || tok == nullptr) {
        fprintf(stderr, "missing embedding or tokenizer role\n");
        return 1;
    }
    if (emb->node_id != "node-a" || tok->node_id != "node-a") {
        fprintf(stderr, "expected tokenizer/embedding on entry boundary node-a, got tok=%s emb=%s\n",
                tok->node_id.c_str(), emb->node_id.c_str());
        return 1;
    }
    return 0;
}

int main() {
    if (test_services_colocated_with_entry_boundary() != 0) {
        fprintf(stderr, "test-embedding-service: FAILED\n");
        return 1;
    }
    printf("test-embedding-service: OK\n");
    return 0;
}
