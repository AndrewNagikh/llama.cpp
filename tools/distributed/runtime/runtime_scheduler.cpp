#include "runtime_scheduler.h"

#include <string>

namespace {

void add_error(runtime_scheduler_validation & out, const std::string & message) {
    out.errors.push_back(message);
}

bool has_runtime_edge(
        const runtime_descriptor & desc,
        const runtime_role from,
        const runtime_role to,
        const std::string & contract) {
    for (const runtime_dependency_descriptor & dependency : desc.dependencies) {
        if (dependency.from == from &&
                dependency.to == to &&
                dependency.data_contract == contract) {
            return true;
        }
    }
    return false;
}

void require_edge(
        const runtime_descriptor & desc,
        const runtime_role from,
        const runtime_role to,
        const std::string & contract,
        runtime_scheduler_validation & out) {
    if (!has_runtime_edge(desc, from, to, contract)) {
        add_error(out,
                "scheduler edge missing: " +
                runtime_role_name(from) +
                " -> " +
                runtime_role_name(to) +
                " (" +
                contract +
                ")");
    }
}

} // namespace

runtime_scheduler_validation validate_runtime_scheduler_operation(
        const runtime_descriptor & desc,
        const runtime_graph & graph,
        const runtime_scheduler_operation op) {
    runtime_scheduler_validation out{};

    const runtime_execution_graph_validation graph_validation =
            validate_runtime_graph_against_descriptor(desc, graph);
    if (!graph_validation.ok()) {
        for (const std::string & error : graph_validation.errors) {
            add_error(out, "graph invalid: " + error);
        }
        out.valid = false;
        return out;
    }

    switch (op) {
        case runtime_scheduler_operation::start_prefill:
        case runtime_scheduler_operation::commit_prefill:
        case runtime_scheduler_operation::start_decode_step:
        case runtime_scheduler_operation::commit_decode_step:
            require_edge(desc, runtime_role::tokenizer, runtime_role::embedding, "token_ids", out);
            require_edge(desc, runtime_role::embedding, runtime_role::pipeline_stage, "hidden_states", out);
            require_edge(desc, runtime_role::pipeline_stage, runtime_role::output_head, "hidden_states", out);
            require_edge(desc, runtime_role::output_head, runtime_role::sampler, "logits", out);
            break;
    }

    if ((op == runtime_scheduler_operation::start_decode_step ||
                op == runtime_scheduler_operation::commit_decode_step) &&
            graph.n_layers <= 0) {
        add_error(out, "decode requires at least one pipeline block");
    }

    out.valid = out.errors.empty();
    return out;
}

runtime_scheduler_validation validate_runtime_scheduler_prefill_decode(
        const runtime_descriptor & desc,
        const runtime_graph & graph) {
    runtime_scheduler_validation out{};
    const runtime_scheduler_operation ops[] = {
        runtime_scheduler_operation::start_prefill,
        runtime_scheduler_operation::commit_prefill,
        runtime_scheduler_operation::start_decode_step,
        runtime_scheduler_operation::commit_decode_step,
    };

    for (const runtime_scheduler_operation op : ops) {
        const runtime_scheduler_validation op_validation =
                validate_runtime_scheduler_operation(desc, graph, op);
        if (!op_validation.ok()) {
            for (const std::string & error : op_validation.errors) {
                add_error(out, error);
            }
        }
    }

    out.valid = out.errors.empty();
    return out;
}
