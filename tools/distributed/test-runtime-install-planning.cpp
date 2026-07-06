#include "runtime/runtime_install_planning.h"

#include "runtime/runtime_descriptor.h"

#include <cstdio>
#include <string>

static int require(const bool condition, const char * message) {
    if (!condition) {
        fprintf(stderr, "test-runtime-install-planning: %s\n", message);
        return 1;
    }
    return 0;
}

static runtime_graph make_graph() {
    runtime_graph graph{};
    graph.model_id = "test-model";
    graph.n_layers = 16;

    runtime_role_assignment tok{};
    tok.role    = runtime_role::tokenizer;
    tok.node_id = "node-b";
    graph.assignments.push_back(tok);

    runtime_role_assignment emb{};
    emb.role    = runtime_role::embedding;
    emb.node_id = "node-a";
    graph.assignments.push_back(emb);

    runtime_role_assignment out{};
    out.role    = runtime_role::output_head;
    out.node_id = "node-c";
    graph.assignments.push_back(out);

    runtime_role_assignment smp{};
    smp.role    = runtime_role::sampler;
    smp.node_id = "node-b";
    graph.assignments.push_back(smp);

    runtime_role_assignment stage_a{};
    stage_a.role        = runtime_role::pipeline_stage;
    stage_a.node_id     = "node-a";
    stage_a.layer_start = 0;
    stage_a.layer_end   = 8;
    graph.assignments.push_back(stage_a);

    runtime_role_assignment stage_c{};
    stage_c.role        = runtime_role::pipeline_stage;
    stage_c.node_id     = "node-c";
    stage_c.layer_start = 8;
    stage_c.layer_end   = 16;
    graph.assignments.push_back(stage_c);

    return graph;
}

static int test_semantic_resource_install_plan() {
    const runtime_descriptor desc =
            make_basic_text_generation_runtime_descriptor("test-model", "llama.arch", 16);
    const runtime_graph graph = make_graph();

    const runtime_resource_install_plan plan =
            build_runtime_resource_install_plan(desc, graph);
    if (!plan.success) {
        fprintf(stderr, "test-runtime-install-planning: semantic install plan failed: %s\n",
                plan.error.c_str());
        return 1;
    }

    if (require(runtime_resource_plan_has(
                    plan, "tokenizer.model", runtime_role::tokenizer, "node-b"),
                "tokenizer resource not placed on tokenizer node") != 0 ||
            require(runtime_resource_plan_has(
                    plan, "embedding.token", runtime_role::embedding, "node-a"),
                "embedding resource not placed on embedding node") != 0 ||
            require(runtime_resource_plan_has(
                    plan, "output.lm_head", runtime_role::output_head, "node-c"),
                "output resource not placed on output node") != 0 ||
            require(runtime_resource_plan_has(
                    plan, "pipeline.blocks", runtime_role::pipeline_stage, "node-a"),
                "pipeline resource not placed on first pipeline node") != 0 ||
            require(runtime_resource_plan_has(
                    plan, "pipeline.blocks", runtime_role::pipeline_stage, "node-c"),
                "pipeline resource not placed on second pipeline node") != 0) {
        return 1;
    }

    const runtime_resource_install_plan validation =
            validate_runtime_resource_install_plan(desc, graph, plan);
    if (require(validation.success, "semantic install plan validation failed") != 0) {
        return 1;
    }
    return 0;
}

static int test_wrong_embedding_node_rejected() {
    const runtime_descriptor desc =
            make_basic_text_generation_runtime_descriptor("test-model", "llama.arch", 16);
    const runtime_graph graph = make_graph();
    runtime_resource_install_plan plan =
            build_runtime_resource_install_plan(desc, graph);
    if (require(plan.success, "base plan failed") != 0) {
        return 1;
    }

    for (runtime_resource_install_assignment & assignment : plan.assignments) {
        if (assignment.resource_id == "embedding.token" &&
                assignment.service_role == runtime_role::embedding) {
            assignment.node_id = "node-b";
        }
    }

    const runtime_resource_install_plan validation =
            validate_runtime_resource_install_plan(desc, graph, plan);
    if (require(!validation.success, "wrong embedding node accepted") != 0) {
        return 1;
    }
    return 0;
}

static int test_wrong_output_node_rejected() {
    const runtime_descriptor desc =
            make_basic_text_generation_runtime_descriptor("test-model", "llama.arch", 16);
    const runtime_graph graph = make_graph();
    runtime_resource_install_plan plan =
            build_runtime_resource_install_plan(desc, graph);
    if (require(plan.success, "base plan failed") != 0) {
        return 1;
    }

    for (runtime_resource_install_assignment & assignment : plan.assignments) {
        if (assignment.resource_id == "output.lm_head" &&
                assignment.service_role == runtime_role::output_head) {
            assignment.node_id = "node-a";
        }
    }

    const runtime_resource_install_plan validation =
            validate_runtime_resource_install_plan(desc, graph, plan);
    if (require(!validation.success, "wrong output node accepted") != 0) {
        return 1;
    }
    return 0;
}

int main() {
    if (test_semantic_resource_install_plan() != 0 ||
            test_wrong_embedding_node_rejected() != 0 ||
            test_wrong_output_node_rejected() != 0) {
        return 1;
    }

    printf("test-runtime-install-planning: OK\n");
    return 0;
}
