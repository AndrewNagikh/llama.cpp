#include "runtime/runtime_descriptor.h"
#include "runtime/runtime_install_planning.h"
#include "runtime/runtime_scheduler.h"

#include <cstdio>
#include <string>
#include <vector>

static int require(const bool condition, const char * message) {
    if (!condition) {
        fprintf(stderr, "test-runtime-acceptance: %s\n", message);
        return 1;
    }
    return 0;
}

static runtime_graph make_concrete_graph(const std::string & model_id, const int32_t n_layers) {
    runtime_graph graph{};
    graph.model_id = model_id;
    graph.n_layers = n_layers;

    runtime_role_assignment tokenizer{};
    tokenizer.role = runtime_role::tokenizer;
    tokenizer.node_id = "node-tokenizer";
    graph.assignments.push_back(tokenizer);

    runtime_role_assignment embedding{};
    embedding.role = runtime_role::embedding;
    embedding.node_id = "node-embedding";
    graph.assignments.push_back(embedding);

    runtime_role_assignment stage_a{};
    stage_a.role = runtime_role::pipeline_stage;
    stage_a.node_id = "node-stage-a";
    stage_a.layer_start = 0;
    stage_a.layer_end = n_layers / 2;
    graph.assignments.push_back(stage_a);

    runtime_role_assignment stage_b{};
    stage_b.role = runtime_role::pipeline_stage;
    stage_b.node_id = "node-stage-b";
    stage_b.layer_start = n_layers / 2;
    stage_b.layer_end = n_layers;
    graph.assignments.push_back(stage_b);

    runtime_role_assignment output{};
    output.role = runtime_role::output_head;
    output.node_id = "node-output";
    graph.assignments.push_back(output);

    runtime_role_assignment sampler{};
    sampler.role = runtime_role::sampler;
    sampler.node_id = "node-sampler";
    graph.assignments.push_back(sampler);

    return graph;
}

static int test_supported_architecture_descriptor_validation() {
    struct architecture_case {
        const char * model_id;
        const char * architecture_id;
        int32_t n_layers;
    };

    const architecture_case cases[] = {
        { "llama-validation", "llama.arch", 32 },
        { "qwen-validation", "qwen.arch", 32 },
        { "gemma-validation", "gemma.arch", 28 },
        { "phi-validation", "phi.arch", 32 },
        { "smollm-validation", "smollm.arch", 24 },
        { "deepseek-validation", "deepseek.arch", 40 },
    };

    for (const architecture_case & item : cases) {
        const runtime_descriptor desc =
                make_basic_text_generation_runtime_descriptor(
                        item.model_id,
                        item.architecture_id,
                        item.n_layers);
        const runtime_descriptor_validation validation = validate_runtime_descriptor(desc);
        if (!validation.ok()) {
            fprintf(stderr,
                    "test-runtime-acceptance: descriptor validation failed for %s: %s\n",
                    item.architecture_id,
                    validation.errors.empty() ? "unknown" : validation.errors.front().c_str());
            return 1;
        }
    }
    return 0;
}

static int test_concrete_graph_acceptance_gates() {
    const runtime_descriptor desc =
            make_basic_text_generation_runtime_descriptor("smoke-candidate", "llama.arch", 12);
    const runtime_graph graph = make_concrete_graph(desc.model_id, 12);

    const runtime_execution_graph_validation graph_validation =
            validate_runtime_graph_against_descriptor(desc, graph);
    if (!graph_validation.ok()) {
        fprintf(stderr,
                "test-runtime-acceptance: concrete graph rejected: %s\n",
                graph_validation.errors.empty() ? "unknown" : graph_validation.errors.front().c_str());
        return 1;
    }

    const runtime_scheduler_validation scheduler_validation =
            validate_runtime_scheduler_prefill_decode(desc, graph);
    if (!scheduler_validation.ok()) {
        fprintf(stderr,
                "test-runtime-acceptance: scheduler legality rejected: %s\n",
                scheduler_validation.errors.empty() ? "unknown" : scheduler_validation.errors.front().c_str());
        return 1;
    }

    const runtime_resource_install_plan install_plan =
            build_runtime_resource_install_plan(desc, graph);
    if (require(install_plan.success, "install readiness failed for concrete graph") != 0) {
        return 1;
    }

    const runtime_resource_install_plan validated_plan =
            validate_runtime_resource_install_plan(desc, graph, install_plan);
    if (require(validated_plan.success, "validated install readiness failed") != 0) {
        return 1;
    }

    return 0;
}

static int test_type_and_pipeline_rejections() {
    runtime_descriptor desc =
            make_basic_text_generation_runtime_descriptor("bad-edge", "llama.arch", 8);
    desc.dependencies[1].data_contract = "logits";
    const runtime_descriptor_validation descriptor_validation =
            validate_runtime_descriptor(desc);
    if (require(!descriptor_validation.ok(), "dependency type mismatch accepted") != 0) {
        return 1;
    }

    const runtime_descriptor valid_desc =
            make_basic_text_generation_runtime_descriptor("bad-graph", "llama.arch", 8);
    runtime_graph graph = make_concrete_graph(valid_desc.model_id, 8);
    graph.assignments[2].layer_end = 3;
    graph.assignments[3].layer_start = 4;

    const runtime_execution_graph_validation graph_validation =
            validate_runtime_graph_against_descriptor(valid_desc, graph);
    if (require(!graph_validation.ok(), "pipeline coverage gap accepted") != 0) {
        return 1;
    }

    const runtime_scheduler_validation scheduler_validation =
            validate_runtime_scheduler_prefill_decode(valid_desc, graph);
    if (require(!scheduler_validation.ok(), "scheduler accepted invalid graph") != 0) {
        return 1;
    }

    return 0;
}

static int test_failure_policy_rejections() {
    runtime_descriptor missing_service_policy =
            make_basic_text_generation_runtime_descriptor("missing-service-policy", "llama.arch", 8);
    missing_service_policy.services[0].failure_policy.clear();
    const runtime_descriptor_validation service_validation =
            validate_runtime_descriptor(missing_service_policy);
    if (require(!service_validation.ok(), "missing service failure policy accepted") != 0) {
        return 1;
    }

    runtime_descriptor missing_edge_policy =
            make_basic_text_generation_runtime_descriptor("missing-edge-policy", "llama.arch", 8);
    missing_edge_policy.dependencies[0].failure_policy.clear();
    const runtime_descriptor_validation edge_validation =
            validate_runtime_descriptor(missing_edge_policy);
    if (require(!edge_validation.ok(), "missing edge failure policy accepted") != 0) {
        return 1;
    }

    return 0;
}

int main() {
    if (test_supported_architecture_descriptor_validation() != 0 ||
            test_concrete_graph_acceptance_gates() != 0 ||
            test_type_and_pipeline_rejections() != 0 ||
            test_failure_policy_rejections() != 0) {
        return 1;
    }

    printf("test-runtime-acceptance: OK\n");
    return 0;
}
