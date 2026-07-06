#include "cluster_consistency.h"

#include "architecture/semantic_runtime_descriptor.h"
#include "runtime/runtime_install_planning.h"

#include <sstream>

namespace {

std::string op_key(const install_operation & op) {
    std::ostringstream os;
    os << install_action_to_string(op.action) << '@' << op.node_id << ':';
    if (!op.download.blob_id.empty()) {
        os << op.download.blob_id << '/' << op.download.tensor_name;
    } else {
        os << "layer=" << op.layer_index;
    }
    return os.str();
}

} // namespace

nlohmann::json install_plan_diff::to_json() const {
    return {
        { "added_ops", added_ops },
        { "removed_ops", removed_ops },
        { "operations", operations },
    };
}

nlohmann::json consistency_check_result::to_json() const {
    return {
        { "consistent", consistent },
        { "idempotent", idempotent },
        { "coverage_ready", coverage_ready },
        { "rebuilt_operation_count", rebuilt_plan.operation_count },
        { "plan_diff", plan_diff.to_json() },
        { "issues", issues },
        { "snapshot", snapshot.to_json() },
    };
}

install_plan_diff diff_install_plan(
        const install_plan & baseline,
        const install_plan & candidate) {
    install_plan_diff diff;

    std::set<std::string> base_keys;
    for (const auto & op : baseline.operations) {
        base_keys.insert(op_key(op));
    }

    for (const auto & op : candidate.operations) {
        const std::string key = op_key(op);
        if (!base_keys.count(key)) {
            diff.added_ops++;
            diff.operations.push_back(op.to_json());
        }
    }

    std::set<std::string> cand_keys;
    for (const auto & op : candidate.operations) {
        cand_keys.insert(op_key(op));
    }
    for (const auto & op : baseline.operations) {
        if (!cand_keys.count(op_key(op))) {
            diff.removed_ops++;
        }
    }

    return diff;
}

bool coverage_fully_ready(const coverage_report & coverage) {
    return coverage.total_layers > 0 &&
           coverage.missing_layers == 0 &&
           coverage.corrupted_layers == 0 &&
           coverage.ready_layers == coverage.total_layers &&
           coverage.state == coverage_state::ready;
}

bool install_plan_is_idempotent(const install_plan & plan) {
    return plan.operation_count == 0 && plan.operations.empty();
}

consistency_check_result check_cluster_consistency(
        const dist_model_record & record,
        const actual_model_layout & actual,
        const std::set<std::string> & online_nodes,
        const std::string & source_url) {
    consistency_check_result result;

    if (!record.layout.has_value() || !record.manifest.has_value()) {
        result.issues.push_back("layout or manifest missing");
        return result;
    }

    const desired_model_layout & desired = record.layout->desired;
    const coverage_report coverage = record.coverage.has_value()
            ? *record.coverage
            : compute_coverage(desired, actual, online_nodes);

    const semantic_runtime_descriptor rt =
            build_semantic_runtime_descriptor(*record.manifest);

    std::optional<runtime_install_node_map> runtime_nodes;
    if (record.stored_runtime_install_nodes.has_value() &&
            record.stored_runtime_install_nodes->valid()) {
        runtime_nodes = *record.stored_runtime_install_nodes;
    } else if (record.stored_runtime_graph.has_value()) {
        runtime_nodes = runtime_install_node_map_from_graph(*record.stored_runtime_graph);
    }
    const runtime_install_node_map * runtime_ptr =
            runtime_nodes.has_value() && runtime_nodes->valid() ? &*runtime_nodes : nullptr;

    const runtime_coverage_report runtime = compute_runtime_coverage(
            rt, desired, actual, online_nodes, runtime_ptr);

    const install_plan stored = record.stored_install_plan.has_value()
            ? *record.stored_install_plan
            : install_plan{};

    result.snapshot = capture_cluster_snapshot(
            record.model_id,
            desired,
            actual,
            coverage,
            runtime,
            stored);

    result.coverage_ready = runtime.fully_ready();

    const auto rebuilt = build_install_plan(
            *record.manifest,
            desired,
            actual,
            coverage,
            source_url,
            runtime_ptr);
    if (!rebuilt.success) {
        result.issues.push_back("rebuild install plan failed: " + rebuilt.error);
        return result;
    }

    result.rebuilt_plan = rebuilt.plan;
    result.idempotent   = install_plan_is_idempotent(rebuilt.plan);

    if (!result.idempotent) {
        result.plan_diff = diff_install_plan(install_plan{}, rebuilt.plan);
        std::ostringstream os;
        os << "non-idempotent install plan: " << rebuilt.plan.operation_count << " operations";
        result.issues.push_back(os.str());
        for (const auto & op : rebuilt.plan.operations) {
            result.issues.push_back(
                    install_action_to_string(op.action) + " " + op.node_id + " " +
                    (op.download.blob_id.empty()
                            ? ("layer=" + std::to_string(op.layer_index))
                            : (op.download.blob_id + "/" + op.download.tensor_name)));
        }
    }

    if (result.coverage_ready && !result.idempotent) {
        result.issues.push_back("coverage READY but install plan not empty");
    }

    if (runtime.misplaced_blobs.empty() &&
            runtime.missing_blobs.empty() &&
            result.coverage_ready &&
            result.idempotent) {
        result.consistent = true;
    } else if (result.coverage_ready && result.idempotent) {
        result.consistent = true;
    } else if (!result.coverage_ready && !result.idempotent) {
        // Expected during partial install — not inconsistent, just incomplete.
        result.consistent = true;
    } else {
        if (!runtime.misplaced_blobs.empty()) {
            result.issues.push_back("misplaced blobs: " +
                    std::to_string(runtime.misplaced_blobs.size()));
        }
        if (!runtime.missing_blobs.empty()) {
            result.issues.push_back("missing blobs: " +
                    std::to_string(runtime.missing_blobs.size()));
        }
    }

    return result;
}

bool assert_coverage_stable(
        const coverage_report & before,
        const coverage_report & after,
        bool store_unchanged,
        std::string & error) {
    if (!store_unchanged) {
        return true;
    }
    if (coverage_fully_ready(before) &&
            !coverage_fully_ready(after) &&
            after.state != coverage_state::degraded) {
        error = "coverage regressed from READY without store change";
        return false;
    }
    if (coverage_fully_ready(before) &&
            before.ready_layers > after.ready_layers &&
            after.missing_layers > before.missing_layers) {
        error = "ready_layers decreased without store change";
        return false;
    }
    return true;
}
