#include "state_snapshot.h"

#include <chrono>
#include <ctime>
#include <iomanip>
#include <sstream>

namespace {

std::string iso_timestamp_now() {
    const auto now = std::chrono::system_clock::now();
    const std::time_t t = std::chrono::system_clock::to_time_t(now);
    std::ostringstream os;
    os << std::put_time(std::gmtime(&t), "%Y-%m-%dT%H:%M:%SZ");
    return os.str();
}

} // namespace

nlohmann::json cluster_state_snapshot::to_json() const {
    return {
        { "model_id", model_id },
        { "captured_at", captured_at },
        { "desired_layout", desired.to_json() },
        { "actual_layout", actual.to_json() },
        { "coverage", coverage.to_json() },
        { "runtime_coverage", runtime.to_json() },
        { "install_plan", plan.to_json() },
        { "layer_store", layer_store_summary },
    };
}

cluster_state_snapshot cluster_state_snapshot::from_json(const nlohmann::json & j) {
    cluster_state_snapshot snap;
    snap.model_id    = j.value("model_id", "");
    snap.captured_at = j.value("captured_at", "");
    if (j.contains("desired_layout")) {
        snap.desired = desired_model_layout::from_json(j["desired_layout"]);
    }
    if (j.contains("actual_layout")) {
        snap.actual = actual_model_layout::from_json(j["actual_layout"]);
    }
    if (j.contains("coverage")) {
        snap.coverage = coverage_report::from_json(j["coverage"]);
    }
    if (j.contains("runtime_coverage")) {
        const auto & rt = j["runtime_coverage"];
        snap.runtime.layer_coverage = coverage_report::from_json(rt);
        if (rt.contains("storage_state")) {
            snap.runtime.storage_state = coverage_state_from_string(rt.value("storage_state", "EMPTY"));
        }
        if (rt.contains("runtime_state")) {
            snap.runtime.runtime_state = coverage_state_from_string(rt.value("runtime_state", "EMPTY"));
        }
        if (rt.contains("missing_blobs") && rt["missing_blobs"].is_array()) {
            for (const auto & item : rt["missing_blobs"]) {
                snap.runtime.missing_blobs.push_back(item.get<std::string>());
            }
        }
        if (rt.contains("misplaced_blobs") && rt["misplaced_blobs"].is_array()) {
            for (const auto & item : rt["misplaced_blobs"]) {
                snap.runtime.misplaced_blobs.push_back(item.get<std::string>());
            }
        }
    }
    if (j.contains("install_plan")) {
        snap.plan = install_plan::from_json(j["install_plan"]);
    }
    if (j.contains("layer_store") && j["layer_store"].is_object()) {
        snap.layer_store_summary = j["layer_store"];
    }
    return snap;
}

cluster_state_snapshot capture_cluster_snapshot(
        const std::string & model_id,
        const desired_model_layout & desired,
        const actual_model_layout & actual,
        const coverage_report & coverage,
        const runtime_coverage_report & runtime,
        const install_plan & plan,
        const nlohmann::json & layer_store_summary) {
    cluster_state_snapshot snap;
    snap.model_id              = model_id;
    snap.captured_at           = iso_timestamp_now();
    snap.desired               = desired;
    snap.actual                = actual;
    snap.coverage              = coverage;
    snap.runtime               = runtime;
    snap.plan                  = plan;
    snap.layer_store_summary   = layer_store_summary;
    return snap;
}

nlohmann::json diff_snapshots(
        const cluster_state_snapshot & before,
        const cluster_state_snapshot & after) {
    nlohmann::json diff = nlohmann::json::object();

    if (before.coverage.state != after.coverage.state) {
        diff["coverage_state"] = {
            { "before", coverage_state_to_string(before.coverage.state) },
            { "after", coverage_state_to_string(after.coverage.state) },
        };
    }
    if (before.coverage.ready_layers != after.coverage.ready_layers ||
            before.coverage.missing_layers != after.coverage.missing_layers) {
        diff["coverage_counts"] = {
            { "before", {
                { "ready", before.coverage.ready_layers },
                { "missing", before.coverage.missing_layers },
            }},
            { "after", {
                { "ready", after.coverage.ready_layers },
                { "missing", after.coverage.missing_layers },
            }},
        };
    }
    if (before.plan.operation_count != after.plan.operation_count) {
        diff["install_plan_ops"] = {
            { "before", before.plan.operation_count },
            { "after", after.plan.operation_count },
        };
    }
    if (before.runtime.misplaced_blobs != after.runtime.misplaced_blobs) {
        diff["misplaced_blobs_changed"] = true;
    }
    return diff;
}
