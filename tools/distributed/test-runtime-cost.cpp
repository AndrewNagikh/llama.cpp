#include "runtime/runtime_cost_model.h"

#include <cstdio>

static runtime_planner_node make_planner_node(
        const char * id,
        int32_t pipeline_layers,
        bool first_stage,
        bool last_stage,
        double cpu_score) {
    runtime_planner_node n{};
    n.node_id          = id;
    n.cpu_score        = cpu_score;
    n.memory_bw_score  = 32.0;
    n.cpu_budget_bytes = 32ULL * 1024ULL * 1024ULL * 1024ULL;
    n.pipeline_layers  = pipeline_layers;
    n.is_first_pipeline_stage = first_stage;
    n.is_last_pipeline_stage  = last_stage;
    return n;
}

static int test_first_stage_penalty() {
    const runtime_role_descriptor desc =
            default_descriptor_for_role(runtime_role::tokenizer, 8ULL * 1024ULL * 1024ULL * 1024ULL);

    const runtime_planner_node first = make_planner_node("first", 8, true, false, 16.0);
    const runtime_planner_node other = make_planner_node("other", 8, false, false, 8.0);

    const double first_cost = runtime_cost_tokenizer(first, desc);
    const double other_cost = runtime_cost_tokenizer(other, desc);
    if (!(other_cost < first_cost)) {
        fprintf(stderr, "test-runtime-cost: first-stage tokenizer cost should exceed other\n");
        return 1;
    }
    return 0;
}

static int test_output_last_stage_penalty() {
    const runtime_role_descriptor desc =
            default_descriptor_for_role(runtime_role::output_head, 8ULL * 1024ULL * 1024ULL * 1024ULL);

    const runtime_planner_node last_node = make_planner_node("last", 8, false, true, 8.0);
    const runtime_planner_node mid_node   = make_planner_node("mid", 8, false, false, 8.0);

    const double last_cost = runtime_cost_output_head(last_node, desc);
    const double mid_cost   = runtime_cost_output_head(mid_node, desc);
    if (!(mid_cost < last_cost)) {
        fprintf(stderr, "test-runtime-cost: output on last stage should cost more\n");
        return 1;
    }
    return 0;
}

int main() {
    if (test_first_stage_penalty() != 0 || test_output_last_stage_penalty() != 0) {
        fprintf(stderr, "test-runtime-cost: FAILED\n");
        return 1;
    }
    printf("test-runtime-cost: OK\n");
    return 0;
}
