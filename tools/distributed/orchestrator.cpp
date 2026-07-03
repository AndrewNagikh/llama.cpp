#include "dist_common.h"
#include "layer_planner.h"
#include "memory_estimator.h"
#include "model_catalog.h"
#include "orchestrator/model_registry.h"
#include "orchestrator/registry_persistence.h"
#include "orchestrator/manifest_builder/manifest_builder.h"
#include "orchestrator/layout_planner/layout_planner.h"
#include "orchestrator/coverage/coverage.h"
#include "orchestrator/coverage/runtime_coverage.h"
#include "architecture/semantic_runtime_descriptor.h"
#include "orchestrator/install_planner/install_planner.h"
#include "orchestrator/optimizer/cluster_optimizer.h"
#include "orchestrator/consistency/cluster_consistency.h"
#include "orchestrator/consistency/state_snapshot.h"
#include "orchestrator/consistency/install_journal.h"
#include "node_agent/layer_store/layer_store.h"
#include "node_agent/layer_store/layer_gguf_assembler.h"
#include "verification/verification_pipeline.h"
#include "verification/worker_verify.h"
#include "split_gen_common.h"
#include "split_tcp_wire.h"

#include "httplib.h"
#include "nlohmann/json.hpp"
#include "llama.h"

#include "ggml-backend.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <map>
#include <mutex>
#include <random>
#include <set>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#if !defined(_WIN32)
#include <unistd.h>
#endif

using json = nlohmann::json;

struct dist_session {
    std::string session_id;
    std::string model;
    std::string model_path;
    std::vector<dist_pipeline_stage> pipeline;
    std::string entry_host;
    int entry_ctrl_port = 0;
    int entry_layer_end = 0;
    bool active         = false;
};

static std::mutex g_mu;
static std::map<std::string, dist_node_info> g_nodes;
static std::map<std::string, dist_session> g_sessions;
static std::string g_model_path;
static std::string g_models_dir;
static int32_t g_n_ctx = 4096;
static model_catalog g_catalog;
static cluster_model_registry g_registry;
static install_journal g_install_journal(std::filesystem::path{});
static std::filesystem::path g_snapshot_dir;
static std::map<std::string, json> g_install_node_results;

struct cluster_sync_job {
    std::string job_id;
    std::string model_id;
    std::string state = "queued";
    std::string error;
    std::map<std::string, std::string> node_job_ids;
    std::map<std::string, json> node_status;
};

static std::map<std::string, cluster_sync_job> g_sync_jobs;
static std::map<std::string, nlohmann::json> g_verify_reports;

static void persist_registry();
static void sync_model_state_from_cluster(const std::string & model_id, bool try_infer_layout);
static void bootstrap_all_models_from_cluster();

static bool poll_installed_layers_from_nodes(
        const std::string & model_id,
        actual_model_layout & out,
        std::set<std::string> & online_nodes);
static std::string resolve_model_source_url(const dist_model_record & record);
static bool build_and_store_install_plan(
        const std::string & model_id,
        dist_model_record * record,
        const desired_model_layout * layout_override = nullptr);

static std::string make_id(const char * prefix) {
    static std::mt19937_64 rng{ std::random_device{}() };
    return std::string(prefix) + "-" + std::to_string(rng());
}

static void set_install_node_result(
        const std::string & job_id,
        const std::string & node_id,
        const json & result) {
    std::lock_guard<std::mutex> lock(g_mu);
    g_install_node_results[job_id][node_id] = result;
}

static void update_install_job(
        const std::string & job_id,
        install_status status,
        const std::string & error_msg = "",
        double progress = 0.0) {
    std::lock_guard<std::mutex> lock(g_mu);
    g_catalog.update_job_status(job_id, status, error_msg, progress);
}

static void complete_install_job(const std::string & job_id) {
    std::lock_guard<std::mutex> lock(g_mu);
    g_catalog.complete_job(job_id);
}

static void coordinate_model_install(
        std::string job_id,
        std::string model_id,
        model_info model,
        std::vector<dist_node_info> nodes) {
    if (nodes.empty()) {
        update_install_job(job_id, install_status::error, "No online nodes available");
        return;
    }

    std::vector<dist_node_info> pending_nodes;
    int ready_nodes = 0;
    int error_nodes = 0;

    for (const auto & node : nodes) {
        httplib::Client client(node.host.c_str(), node.http_port);
        client.set_connection_timeout(5, 0);
        client.set_read_timeout(30, 0);
        client.set_write_timeout(30, 0);

        json install_request = {
            { "model", model_id },
            { "repo", model.source.repo },
            { "file", model.source.file }
        };

        const auto result = client.Post("/models/install", install_request.dump(), "application/json");
        if (!result || result->status != 200) {
            ++error_nodes;
            set_install_node_result(job_id, node.node_id, {
                { "status", "error" },
                { "error", result ? result->body : "node unreachable" }
            });
            continue;
        }

        json body;
        try {
            body = json::parse(result->body);
        } catch (...) {
            ++error_nodes;
            set_install_node_result(job_id, node.node_id, {
                { "status", "error" },
                { "error", "invalid node response" }
            });
            continue;
        }

        const std::string status = body.value("status", "download_started");
        const std::string local_path = body.value("local_path", "");
        set_install_node_result(job_id, node.node_id, {
            { "status", status },
            { "local_path", local_path }
        });

        if (status == "already_installed") {
            ++ready_nodes;
        } else {
            pending_nodes.push_back(node);
        }
    }

    const int total_nodes = static_cast<int>(nodes.size());
    if (ready_nodes + error_nodes == total_nodes) {
        if (error_nodes == 0) {
            complete_install_job(job_id);
        } else {
            update_install_job(job_id, install_status::error, "Installation failed on all available nodes");
        }
        return;
    }

    update_install_job(job_id, install_status::downloading, "",
            static_cast<double>(ready_nodes) / static_cast<double>(total_nodes));

    constexpr int max_attempts = 1800; // Up to 1 hour at 2 seconds per poll.
    for (int attempt = 0; attempt < max_attempts && !pending_nodes.empty(); ++attempt) {
        std::this_thread::sleep_for(std::chrono::seconds(2));

        std::vector<dist_node_info> still_pending;
        for (const auto & node : pending_nodes) {
            httplib::Client client(node.host.c_str(), node.http_port);
            client.set_connection_timeout(5, 0);
            client.set_read_timeout(10, 0);

            const auto result = client.Get("/models/local");
            if (!result || result->status != 200) {
                still_pending.push_back(node);
                continue;
            }

            bool found = false;
            bool done = false;
            try {
                const json models = json::parse(result->body);
                for (const auto & item : models) {
                    if (item.value("model_id", "") != model_id) {
                        continue;
                    }

                    found = true;
                    const std::string status = item.value("status", item.value("ready", false) ? "ready" : "unknown");
                    if (status == "ready" && item.value("ready", false)) {
                        ++ready_nodes;
                        done = true;
                        set_install_node_result(job_id, node.node_id, {
                            { "status", "ready" },
                            { "local_path", item.value("local_path", "") },
                            { "size_bytes", item.value("size_bytes", static_cast<uint64_t>(0)) }
                        });
                    } else if (status == "error") {
                        ++error_nodes;
                        done = true;
                        set_install_node_result(job_id, node.node_id, {
                            { "status", "error" },
                            { "local_path", item.value("local_path", "") },
                            { "error", item.value("error", "download failed") }
                        });
                    }
                    break;
                }
            } catch (...) {
                // Treat transient parse errors as still pending.
            }

            if (!found || !done) {
                still_pending.push_back(node);
            }
        }

        pending_nodes = std::move(still_pending);
        update_install_job(job_id, install_status::downloading, "",
                static_cast<double>(ready_nodes) / static_cast<double>(total_nodes));

        if (ready_nodes + error_nodes == total_nodes) {
            break;
        }
    }

    if (ready_nodes == total_nodes) {
        complete_install_job(job_id);
    } else {
        update_install_job(job_id, install_status::error,
                "Installation incomplete (" + std::to_string(ready_nodes) + "/" +
                std::to_string(total_nodes) + " nodes ready)",
                static_cast<double>(ready_nodes) / static_cast<double>(total_nodes));
    }
}

static void set_sync_job_state(
        const std::string & job_id,
        const std::string & state,
        const std::string & error = "") {
    std::lock_guard<std::mutex> lock(g_mu);
    auto it = g_sync_jobs.find(job_id);
    if (it == g_sync_jobs.end()) {
        return;
    }
    it->second.state = state;
    if (!error.empty()) {
        it->second.error = error;
    }
}

static void coordinate_install_plan_execute(
        std::string job_id,
        std::string model_id,
        install_plan plan,
        model_manifest manifest,
        std::map<std::string, dist_node_info> nodes) {
    if (nodes.empty()) {
        set_sync_job_state(job_id, "failed", "no online nodes");
        return;
    }

    if (plan.operations.empty()) {
        set_sync_job_state(job_id, "completed");
        return;
    }

    std::map<std::string, std::vector<install_operation>> by_node;
    for (const auto & op : plan.operations) {
        by_node[op.node_id].push_back(op);
    }

    int dispatched = 0;
    int failed_dispatch = 0;

    for (const auto & kv : by_node) {
        const auto node_it = nodes.find(kv.first);
        if (node_it == nodes.end()) {
            ++failed_dispatch;
            continue;
        }
        const dist_node_info & node = node_it->second;

        json body = {
            { "operations", json::array() },
            { "manifest", manifest.to_json() },
        };
        for (const auto & op : kv.second) {
            body["operations"].push_back(op.to_json());
        }

        httplib::Client client(node.host.c_str(), node.http_port);
        client.set_connection_timeout(5, 0);
        client.set_read_timeout(120, 0);
        client.set_write_timeout(120, 0);

        const std::string path = "/models/" + model_id + "/install/execute";
        const auto result = client.Post(path.c_str(), body.dump(), "application/json");
        if (!result || result->status != 200) {
            ++failed_dispatch;
            std::lock_guard<std::mutex> lock(g_mu);
            g_sync_jobs[job_id].node_status[kv.first] = {
                { "status", "error" },
                { "error", result ? result->body : "node unreachable" },
            };
            continue;
        }

        try {
            const json resp = json::parse(result->body);
            const std::string node_job_id = resp.value("job_id", "");
            {
                std::lock_guard<std::mutex> lock(g_mu);
                g_sync_jobs[job_id].node_job_ids[kv.first] = node_job_id;
                g_sync_jobs[job_id].node_status[kv.first] = {
                    { "status", "running" },
                    { "job_id", node_job_id },
                };
            }
            ++dispatched;
        } catch (...) {
            ++failed_dispatch;
        }
    }

    if (dispatched == 0) {
        set_sync_job_state(job_id, "failed", "failed to dispatch to any node");
        return;
    }

    set_sync_job_state(job_id, "running");

    constexpr int max_attempts = 1800;
    for (int attempt = 0; attempt < max_attempts; ++attempt) {
        std::this_thread::sleep_for(std::chrono::seconds(1));

        int completed_nodes = 0;
        int failed_nodes    = 0;

        std::map<std::string, std::string> node_jobs;
        {
            std::lock_guard<std::mutex> lock(g_mu);
            node_jobs = g_sync_jobs[job_id].node_job_ids;
        }

        for (const auto & nj : node_jobs) {
            const auto node_it = nodes.find(nj.first);
            if (node_it == nodes.end()) {
                ++failed_nodes;
                continue;
            }

            httplib::Client client(node_it->second.host.c_str(), node_it->second.http_port);
            client.set_connection_timeout(3, 0);
            client.set_read_timeout(10, 0);

            const auto result = client.Get(("/jobs/" + nj.second).c_str());
            if (!result || result->status != 200) {
                continue;
            }

            try {
                const json job = json::parse(result->body);
                const std::string st = job.value("state", "");
                {
                    std::lock_guard<std::mutex> lock(g_mu);
                    g_sync_jobs[job_id].node_status[nj.first] = job;
                }
                if (st == "COMPLETED") {
                    ++completed_nodes;
                } else if (st == "FAILED") {
                    ++failed_nodes;
                }
            } catch (...) {
                // keep polling
            }
        }

        if (completed_nodes + failed_nodes == static_cast<int>(node_jobs.size())) {
            if (failed_nodes == 0) {
                set_sync_job_state(job_id, "completed");
                actual_model_layout actual;
                std::set<std::string> online_nodes;
                poll_installed_layers_from_nodes(model_id, actual, online_nodes);
                dist_model_record * record = g_registry.find(model_id);
                if (record) {
                    g_registry.apply_actual(model_id, actual, record);
                    g_registry.refresh_coverage(model_id, online_nodes, record);
                    record = g_registry.find(model_id);
                    if (record) {
                        build_and_store_install_plan(model_id, record);
                        record = g_registry.find(model_id);
                        if (record && record->coverage.has_value() &&
                                record->stored_install_plan.has_value()) {
                            const consistency_check_result check =
                                    check_cluster_consistency(
                                            *record,
                                            actual,
                                            online_nodes,
                                            resolve_model_source_url(*record));
                            if (!check.idempotent) {
                                fprintf(stderr,
                                        "orchestrator: post-install not idempotent model=%s ops=%d\n",
                                        model_id.c_str(),
                                        check.rebuilt_plan.operation_count);
                                for (const auto & issue : check.issues) {
                                    fprintf(stderr, "  %s\n", issue.c_str());
                                }
                            }
                            if (!g_snapshot_dir.empty()) {
                                std::error_code ec;
                                std::filesystem::create_directories(g_snapshot_dir, ec);
                                const auto path = g_snapshot_dir /
                                        (model_id + "-" + job_id + ".json");
                                std::ofstream out(path);
                                if (out) {
                                    out << check.snapshot.to_json().dump(2);
                                }
                            }
                        }
                    }
                    persist_registry();
                }
            } else {
                set_sync_job_state(job_id, "failed", "one or more node jobs failed");
            }
            return;
        }
    }

    set_sync_job_state(job_id, "failed", "sync job timed out");
}

static std::map<std::string, dist_node_info> collect_online_nodes_copy() {
    std::map<std::string, dist_node_info> nodes_copy;
    std::lock_guard<std::mutex> lock(g_mu);
    for (const auto & kv : g_nodes) {
        if (kv.second.online) {
            nodes_copy[kv.first] = kv.second;
        }
    }
    return nodes_copy;
}

static std::string resolve_model_source_url(const dist_model_record & record) {
    std::string source_url;

    for (const auto & file : record.files) {
        if (!file.download_url.empty() &&
                (record.filename.empty() || file.filename == record.filename)) {
            source_url = file.download_url;
            break;
        }
    }

    if (source_url.empty() && record.manifest.has_value() &&
            !record.manifest->source_file.empty()) {
        std::error_code ec;
        if (std::filesystem::exists(record.manifest->source_file, ec)) {
            source_url = "file://" + record.manifest->source_file;
        }
    }

    if (source_url.empty() && !g_models_dir.empty() && !record.filename.empty()) {
        const std::string candidate =
                (std::filesystem::path(g_models_dir) / record.filename).string();
        std::error_code ec;
        if (std::filesystem::exists(candidate, ec)) {
            source_url = "file://" + candidate;
        }
    }

    if (source_url.empty() && !record.repository.empty() && !record.filename.empty()) {
        const std::string revision = record.revision.empty() ? "main" : record.revision;
        source_url = "https://huggingface.co/" + record.repository +
                     "/resolve/" + revision + "/" + record.filename;
    }

    return source_url;
}

static const desired_model_layout * planning_target_layout(const dist_model_record & record) {
    if (record.pending_layout.has_value()) {
        return &record.pending_layout->desired;
    }
    if (record.layout.has_value()) {
        return &record.layout->desired;
    }
    return nullptr;
}

static bool build_and_store_install_plan(
        const std::string & model_id,
        dist_model_record * record,
        const desired_model_layout * layout_override) {
    if (!record || !record->manifest.has_value()) {
        return false;
    }

    const desired_model_layout * target = layout_override;
    if (!target) {
        // Use committed layout for normal installs. Optimizer pending_layout is
        // only applied during an explicit rebalance pipeline.
        if (record->layout.has_value()) {
            target = &record->layout->desired;
        } else if (record->pending_layout.has_value()) {
            target = &record->pending_layout->desired;
        }
    }
    if (!target) {
        return false;
    }

    actual_model_layout actual;
    std::set<std::string> online_nodes;
    poll_installed_layers_from_nodes(model_id, actual, online_nodes);
    g_registry.apply_actual(model_id, actual, record);
    g_registry.refresh_coverage(model_id, online_nodes, record);

    record = g_registry.find(model_id);
    if (!record || !record->coverage.has_value()) {
        return false;
    }

    const coverage_report & coverage_for_plan = layout_override
            ? compute_coverage(*layout_override, actual, online_nodes)
            : *record->coverage;

    const auto built = build_install_plan(
            *record->manifest,
            *target,
            actual,
            coverage_for_plan,
            resolve_model_source_url(*record));
    if (!built.success) {
        return false;
    }

    std::string verr;
    if (!validate_install_plan(built.plan, *target, coverage_for_plan, verr)) {
        fprintf(stderr, "orchestrator: install plan validation failed for %s: %s\n",
                model_id.c_str(), verr.c_str());
        return false;
    }

    const bool applied = g_registry.apply_install_plan(model_id, built.plan, record);
    if (applied) {
        persist_registry();
    }
    return applied;
}

static optimization_result optimize_registered_model(
        const std::string & model_id,
        const optimizer_policy * policy_override = nullptr) {
    optimization_result empty;
    empty.model_id = model_id;

    dist_model_record * record = g_registry.find(model_id);
    if (!record || !record->manifest.has_value()) {
        empty.reason = "model or manifest not ready";
        return empty;
    }

    const desired_model_layout * current = nullptr;
    if (record->layout.has_value()) {
        current = &record->layout->desired;
    }

    std::vector<dist_node_info> node_vec;
    const auto nodes_copy = collect_online_nodes_copy();
    node_vec.reserve(nodes_copy.size());
    for (const auto & kv : nodes_copy) {
        node_vec.push_back(kv.second);
    }

    const optimizer_policy policy = policy_override ? *policy_override : optimizer_policy{};
    optimization_result result = run_cluster_optimization(
            model_id,
            *record->manifest,
            current,
            node_vec,
            policy,
            g_n_ctx);

    g_registry.apply_optimization(model_id, result);
    return result;
}

static void coordinate_rebalance_pipeline(const std::string model_id) {
    dist_model_record * record = g_registry.find(model_id);
    if (!record || !record->optimization.has_value()) {
        return;
    }
    if (record->optimization->decision != optimizer_decision::rebalance) {
        return;
    }

    g_registry.apply_pending_layout(model_id, record->optimization->candidate_layout);
    record = g_registry.find(model_id);
    if (!record) {
        return;
    }

    if (!build_and_store_install_plan(model_id, record, &record->pending_layout->desired)) {
        g_registry.discard_pending_layout(model_id);
        return;
    }

    record = g_registry.find(model_id);
    if (!record || !record->stored_install_plan.has_value() || !record->manifest.has_value()) {
        g_registry.discard_pending_layout(model_id);
        return;
    }

    const std::string job_id = make_id("job");
  {
        std::lock_guard<std::mutex> lock(g_mu);
        cluster_sync_job job;
        job.job_id   = job_id;
        job.model_id = model_id;
        job.state    = "queued";
        g_sync_jobs[job_id] = std::move(job);
    }

    const auto nodes_copy = collect_online_nodes_copy();
    coordinate_install_plan_execute(
            job_id,
            model_id,
            *record->stored_install_plan,
            *record->manifest,
            nodes_copy);

    {
        std::lock_guard<std::mutex> lock(g_mu);
        const auto it = g_sync_jobs.find(job_id);
        if (it == g_sync_jobs.end() || it->second.state != "completed") {
            g_registry.discard_pending_layout(model_id);
            return;
        }
    }

    actual_model_layout actual;
    std::set<std::string> online_nodes;
    poll_installed_layers_from_nodes(model_id, actual, online_nodes);
    g_registry.apply_actual(model_id, actual);
    g_registry.refresh_pending_coverage(model_id, online_nodes);

    record = g_registry.find(model_id);
    if (!record || !record->coverage.has_value() ||
            record->coverage->state != coverage_state::ready) {
        g_registry.discard_pending_layout(model_id);
        return;
    }

    g_registry.commit_pending_layout(model_id);
}

static void persist_registry() {
    if (g_models_dir.empty()) {
        return;
    }
    registry_persistence_save(g_models_dir, g_registry);
}

static void sync_model_state_from_cluster(
        const std::string & model_id,
        const bool try_infer_layout) {
    dist_model_record * record = g_registry.find(model_id);
    if (!record || !record->manifest.has_value()) {
        return;
    }

    actual_model_layout actual;
    std::set<std::string> online_nodes;
    poll_installed_layers_from_nodes(model_id, actual, online_nodes);
    g_registry.apply_actual(model_id, actual, record);

    record = g_registry.find(model_id);
    if (!record) {
        return;
    }

    const int32_t n_layer = static_cast<int32_t>(record->manifest->n_layer);
    const bool needs_layout = !record->layout.has_value() ||
            record->layout->desired.placements.empty();

    if (try_infer_layout && needs_layout && n_layer > 0) {
        if (const auto inferred = desired_layout_from_actual(model_id, actual, n_layer)) {
            g_registry.apply_layout(model_id, *inferred, record);
            record = g_registry.find(model_id);
            fprintf(stderr,
                    "orchestrator: inferred layout for %s from installed layers (%d placements)\n",
                    model_id.c_str(),
                    static_cast<int>(inferred->placements.size()));
        }
    }

    if (record && record->layout.has_value() && !record->layout->desired.placements.empty()) {
        g_registry.refresh_coverage(model_id, online_nodes, record);
        record = g_registry.find(model_id);

        // Saved layout may be stale after restart; adopt on-disk placement when fully installed.
        if (try_infer_layout && record && record->coverage.has_value() &&
                record->coverage->state != coverage_state::ready && n_layer > 0) {
            if (const auto inferred = desired_layout_from_actual(model_id, actual, n_layer)) {
                if (!layouts_placement_equal(record->layout->desired, *inferred)) {
                    g_registry.apply_layout(model_id, *inferred, record);
                    record = g_registry.find(model_id);
                    fprintf(stderr,
                            "orchestrator: adopted actual layout for %s (%d placements, saved layout mismatch)\n",
                            model_id.c_str(),
                            static_cast<int>(inferred->placements.size()));
                    if (record) {
                        g_registry.refresh_coverage(model_id, online_nodes, record);
                    }
                }
            }
        }
    } else if (record && record->layout.has_value()) {
        g_registry.refresh_coverage(model_id, online_nodes, record);
    }
}

static void bootstrap_all_models_from_cluster() {
    for (const auto & model : g_registry.list()) {
        if (model.manifest.has_value()) {
            sync_model_state_from_cluster(model.model_id, true);
        }
    }
    persist_registry();
}

static void trigger_cluster_optimization_async() {
    std::thread([]() {
        const auto models = g_registry.list();
        for (const auto & model : models) {
            if (!model.manifest.has_value()) {
                continue;
            }
            if (model.coverage.has_value() &&
                    model.coverage->state == coverage_state::ready) {
                continue;
            }
            const optimization_result result = optimize_registered_model(model.model_id);
            if (result.decision == optimizer_decision::rebalance) {
                coordinate_rebalance_pipeline(model.model_id);
            }
        }
    }).detach();
}

static bool gen3_send_recv(
        int ctrl_fd,
        split_gen_cmd cmd,
        int32_t n_tokens,
        int32_t pos_start,
        int32_t layer_end,
        const int32_t * tokens,
        split_gen3_a_resp & resp) {
    if (!split_gen_send_req(ctrl_fd, cmd, n_tokens, pos_start, layer_end, 0, tokens)) {
        return false;
    }
    return split_gen3_recv_a_resp(ctrl_fd, resp, nullptr);
}

static bool configure_node(
        const dist_node_info & node,
        const json & body,
        std::string & err,
        int timeout_ms = 30000) {
    httplib::Client cli(node.host.c_str(), node.http_port);
    cli.set_connection_timeout(5, 0);
    cli.set_read_timeout(timeout_ms / 1000, (timeout_ms % 1000) * 1000);

    const auto res = cli.Post("/configure", body.dump(), "application/json");
    if (!res) {
        err = "no response from " + node.node_id + " at " + node.host + ":" + std::to_string(node.http_port);
        return false;
    }
    if (res->status != 200) {
        err = node.node_id + " configure HTTP " + std::to_string(res->status) + ": " + res->body;
        return false;
    }

    try {
        const json j = json::parse(res->body);
        if (!j.value("ok", false)) {
            err = node.node_id + ": " + j.value("error", "configure failed");
            return false;
        }
        return true;
    } catch (...) {
        err = node.node_id + ": invalid configure response";
        return false;
    }
}

static bool shutdown_node(const dist_node_info & node) {
    httplib::Client cli(node.host.c_str(), node.http_port);
    cli.set_connection_timeout(2, 0);
    cli.set_read_timeout(5, 0);
    const auto res = cli.Post("/shutdown", "", "application/json");
    return res && res->status == 200;
}

static bool check_node_health(const dist_node_info & node, std::string & err) {
    httplib::Client cli(node.host.c_str(), node.http_port);
    cli.set_connection_timeout(3, 0);
    cli.set_read_timeout(3, 0);
    const auto res = cli.Get("/health");
    if (!res) {
        err = node.node_id + " unreachable at " + node.host + ":" + std::to_string(node.http_port);
        return false;
    }
    if (res->status != 200) {
        err = node.node_id + " health check failed HTTP " + std::to_string(res->status);
        return false;
    }
    return true;
}

static json planned_layout_json(const std::vector<dist_layer_assignment> & assignments) {
    json layout = json::array();
    const char * roles[] = { "entry", "middle", "final" };
    for (size_t i = 0; i < assignments.size(); ++i) {
        const char * role = "middle";
        if (i == 0) {
            role = "entry";
        } else if (i + 1 == assignments.size()) {
            role = "final";
        } else if (assignments.size() == 1) {
            role = "entry";
        }
        layout.push_back({
            { "node", assignments[i].node_id },
            { "start", assignments[i].layer_start },
            { "end", assignments[i].layer_end },
            { "score", assignments[i].score },
            { "role", role },
        });
    }
    return layout;
}

static std::vector<dist_layer_assignment> assignments_from_desired_layout(
        const desired_model_layout & desired,
        const std::map<std::string, dist_node_info> & node_map,
        int n_layers) {
    std::vector<dist_layer_assignment> out;
    if (desired.placements.empty() || n_layers <= 0) {
        return out;
    }

    std::vector<layer_placement> sorted = desired.placements;
    std::sort(sorted.begin(), sorted.end(),
            [](const layer_placement & a, const layer_placement & b) {
                return a.layer_index < b.layer_index;
            });

    dist_layer_assignment current{};
    for (const auto & placement : sorted) {
        if (placement.layer_index < 0 || placement.layer_index >= n_layers) {
            continue;
        }
        const auto node_it = node_map.find(placement.node_id);
        if (node_it == node_map.end() || !node_it->second.online) {
            return {};
        }

        if (current.node_id.empty()) {
            current.node_id     = placement.node_id;
            current.score       = node_it->second.score;
            current.layer_start = placement.layer_index;
            current.layer_end   = placement.layer_index + 1;
            continue;
        }

        if (placement.node_id == current.node_id && placement.layer_index == current.layer_end) {
            current.layer_end = placement.layer_index + 1;
            continue;
        }

        out.push_back(current);
        current = {};
        current.node_id     = placement.node_id;
        current.score       = node_it->second.score;
        current.layer_start = placement.layer_index;
        current.layer_end   = placement.layer_index + 1;
    }

    if (!current.node_id.empty()) {
        out.push_back(current);
    }
    return out;
}

static std::string resolve_model_path(const dist_model_record & record) {
    if (!record.filename.empty()) {
        if (!g_model_path.empty() &&
            std::filesystem::path(g_model_path).filename() == record.filename &&
            std::filesystem::exists(g_model_path)) {
            return g_model_path;
        }
        if (!g_models_dir.empty()) {
            const std::string candidate =
                (std::filesystem::path(g_models_dir) / record.filename).string();
            if (std::filesystem::exists(candidate)) {
                return candidate;
            }
        }
    }

    return {};
}

static model_memory_requirements get_model_memory_for_record(
        const dist_model_record & record,
        int32_t                   n_ctx) {
    if (record.manifest.has_value() && !record.manifest->empty()) {
        model_memory_requirements mem = estimate_model_memory_from_manifest(*record.manifest, n_ctx);
        if (mem.valid()) {
            mem.model_id = record.model_id;
            return mem;
        }
    }

    // Prefer a local GGUF matching the registry filename.
    const std::string path = resolve_model_path(record);
    if (!path.empty()) {
        model_memory_requirements mem = estimate_model_memory(path, n_ctx);
        if (mem.valid()) {
            mem.model_id = record.model_id;
            return mem;
        }
    }

    // Otherwise fall back to the legacy catalog entry.
    {
        std::lock_guard<std::mutex> lock(g_mu);
        const auto * model = g_catalog.find_model(record.model_id);
        if (model) {
            return estimate_model_memory_from_catalog(*model, n_ctx);
        }
    }

    return {};
}

static std::string default_models_dir() {
    const char * home = std::getenv("HOME");
    if (home != nullptr && home[0] != '\0') {
        return std::string(home) + "/.distributed-llm/models";
    }
    return ".distributed-llm/models";
}

static bool cache_manifest_metadata(const std::string & model_id, const model_manifest & manifest) {
    if (g_models_dir.empty() || manifest.tensor_data_offset == 0) {
        return false;
    }
    dist_model_record * record = g_registry.find(model_id);
    if (!record) {
        return false;
    }
    const std::string source_url = resolve_model_source_url(*record);
    if (source_url.empty()) {
        return false;
    }
    layer_store store(g_models_dir, model_id);
    return layer_store_cache_metadata(store, manifest, source_url);
}

static std::string layout_entry_node_id(const dist_model_record & record) {
    if (!record.layout.has_value()) {
        return {};
    }
    for (const auto & placement : record.layout->desired.placements) {
        if (placement.layer_index == 0) {
            return placement.node_id;
        }
    }
    int32_t min_layer = INT32_MAX;
    std::string node_id;
    for (const auto & placement : record.layout->desired.placements) {
        if (placement.layer_index < min_layer) {
            min_layer = placement.layer_index;
            node_id   = placement.node_id;
        }
    }
    return node_id;
}

static std::string fetch_entry_worker_tokenizer(
        const dist_model_record & record,
        const std::string & dest_path) {
    const std::string node_id = layout_entry_node_id(record);
    if (node_id.empty()) {
        return {};
    }

    dist_node_info node{};
    {
        std::lock_guard<std::mutex> lock(g_mu);
        const auto it = g_nodes.find(node_id);
        if (it == g_nodes.end() || !it->second.online) {
            return {};
        }
        node = it->second;
    }

    httplib::Client client(node.host.c_str(), node.http_port);
    client.set_connection_timeout(5, 0);
    client.set_read_timeout(600, 0);
    const std::string path = "/models/" + record.model_id + "/workers/entry";

    std::ofstream out(dest_path, std::ios::binary | std::ios::trunc);
    if (!out) {
        return {};
    }

    bool write_ok = true;
    const auto result = client.Get(path.c_str(), [&](const char * data, size_t len) {
        if (!write_ok || len == 0) {
            return write_ok;
        }
        out.write(data, static_cast<std::streamsize>(len));
        if (!out) {
            write_ok = false;
        }
        return write_ok;
    });
    out.close();

    if (!result || result->status != 200 || !write_ok) {
        std::error_code ec;
        std::filesystem::remove(dest_path, ec);
        return {};
    }

    std::error_code ec;
    if (!std::filesystem::exists(dest_path, ec) ||
            std::filesystem::file_size(dest_path, ec) == 0) {
        std::filesystem::remove(dest_path, ec);
        return {};
    }
    return dest_path;
}

static std::string resolve_tokenizer_gguf_path(const dist_model_record & record) {
    const std::string local = resolve_model_path(record);
    if (!local.empty()) {
        return local;
    }
    if (!record.manifest.has_value() || record.manifest->empty()) {
        return {};
    }

    if (g_models_dir.empty()) {
        return {};
    }

    layer_store store(g_models_dir, record.model_id);
    if (!store.metadata_bytes().has_value()) {
        cache_manifest_metadata(record.model_id, *record.manifest);
    }

    const std::string out =
            (store.model_root() / "tokenizer.gguf").string();
    std::error_code ec;
    if (std::filesystem::exists(out, ec)) {
        const uint64_t meta_size = record.manifest->tensor_data_offset;
        const auto fsize = std::filesystem::file_size(out, ec);
        if (!ec && fsize > meta_size) {
            return out;
        }
        std::filesystem::remove(out, ec);
    }

    if (const std::string fetched = fetch_entry_worker_tokenizer(record, out); !fetched.empty()) {
        return fetched;
    }

    if (layer_store_materialize_gguf(
                store,
                *record.manifest,
                out,
                0,
                0,
                false,
                false)) {
        return out;
    }
    return {};
}

static void dist_print_planner_report(
        const std::string & model_id,
        const model_memory_requirements & mem,
        const std::vector<dist_node_info> & nodes,
        const cluster_memory_fits_result & fit,
        const std::vector<dist_layer_assignment> & layout) {
    fprintf(stderr, "\nPlanner Report\n");
    fprintf(stderr, "Model: %s\n", model_id.empty() ? "unknown" : model_id.c_str());
    fprintf(stderr, "Weights: %.1f GB\n", mem.weights_gb());
    fprintf(stderr, "Estimated KV: %.1f GB\n", mem.kv_gb());
    fprintf(stderr, "Compute: %.1f GB\n", mem.compute_gb());
    fprintf(stderr, "Scratch: %.1f GB\n", mem.scratch_gb());
    fprintf(stderr, "Required: %.1f GB\n", mem.total_gb());
    fprintf(stderr, "Cluster:\n");
    for (const auto & n : nodes) {
        if (!n.online) {
            continue;
        }
        const char * kind = n.memory.has_gpu ? "GPU" : "CPU";
        const char * backend = n.caps.gpu_backend.empty() ? "cpu" : n.caps.gpu_backend.c_str();
        fprintf(stderr, "  Node %s  %s/%s  free VRAM %.1f GB  free RAM %.1f GB\n",
                n.node_id.c_str(), backend, kind,
                dist_bytes_to_gb(n.memory.free_vram_bytes),
                dist_bytes_to_gb(n.memory.free_ram_bytes));
    }
    fprintf(stderr, "Result Fits: %s\n", fit.fits ? "YES" : "NO");
    if (fit.fits) {
        fprintf(stderr, "Layer layout\n");
        for (const auto & a : layout) {
            fprintf(stderr, "  Node %s %d-%d\n",
                    a.node_id.c_str(), a.layer_start, a.layer_end);
        }
    }
    fprintf(stderr, "\n");
}

static const dist_node_info * find_node(const std::map<std::string, dist_node_info> & nodes, const std::string & id) {
    const auto it = nodes.find(id);
    return it == nodes.end() ? nullptr : &it->second;
}

static bool poll_installed_layers_from_nodes(
        const std::string & model_id,
        actual_model_layout & out,
        std::set<std::string> & online_nodes) {
    std::vector<actual_model_layout> reports;
    online_nodes.clear();

    std::map<std::string, dist_node_info> nodes_copy;
    {
        std::lock_guard<std::mutex> lock(g_mu);
        nodes_copy = g_nodes;
    }

    for (const auto & kv : nodes_copy) {
        if (!kv.second.online) {
            continue;
        }

        httplib::Client client(kv.second.host.c_str(), kv.second.http_port);
        client.set_connection_timeout(3, 0);
        client.set_read_timeout(10, 0);

        const std::string path = "/installed-layers?model=" + model_id;
        const auto result = client.Get(path.c_str());
        if (!result || result->status != 200) {
            continue;
        }

        try {
            const json body = json::parse(result->body);
            reports.push_back(actual_layout_from_node_response(kv.second.node_id, body));
            online_nodes.insert(kv.second.node_id);
        } catch (...) {
            continue;
        }
    }

    out = merge_node_layer_reports(model_id, reports);
    return true;
}

static bool setup_pipeline(
        const std::string & session_id,
        int n_layers,
        const std::vector<dist_layer_assignment> & assignments,
        const std::map<std::string, dist_node_info> & node_map,
        dist_session & session,
        std::string & err) {
    if (assignments.empty()) {
        err = "empty layer plan";
        return false;
    }

    const int pipe_base = 9100 + (int) (getpid() % 500) + 10;

    std::vector<dist_pipeline_stage> stages;
    stages.reserve(assignments.size() < 2 ? 2 : assignments.size());

    auto push_stage = [&](const dist_layer_assignment & assign, const size_t i, const size_t n) {
        const dist_node_info * node = find_node(node_map, assign.node_id);
        if (!node) {
            err = "unknown node in plan: " + assign.node_id;
            return false;
        }

        dist_pipeline_stage stage{};
        stage.node_id     = node->node_id;
        stage.host        = node->host;
        stage.http_port   = node->http_port;
        stage.layer_start = assign.layer_start;
        stage.layer_end   = assign.layer_end;
        stage.score       = assign.score;

        if (i == 0) {
            stage.role      = DIST_ROLE_ENTRY;
            stage.ctrl_port = pipe_base + 1;
        } else if (i + 1 == n) {
            stage.role      = DIST_ROLE_FINAL;
            stage.peer_port = pipe_base + (int) i + 1;
        } else {
            stage.role      = DIST_ROLE_MIDDLE;
            stage.peer_port = pipe_base + (int) i + 1;
        }

        stages.push_back(stage);
        return true;
    };

    if (assignments.size() == 1) {
        // One node holds all layers: split into entry + final stages on the same host.
        const dist_layer_assignment & full = assignments[0];
        const int total = full.layer_end - full.layer_start;
        if (total < 2) {
            err = "model too small for 2-stage pipeline";
            return false;
        }
        const int split = full.layer_start + total / 2;

        dist_layer_assignment entry = full;
        entry.layer_end = split;

        dist_layer_assignment final = full;
        final.layer_start = split;

        if (!push_stage(entry, 0, 2)) {
            return false;
        }
        if (!push_stage(final, 1, 2)) {
            return false;
        }
    } else {
        for (size_t i = 0; i < assignments.size(); ++i) {
            if (!push_stage(assignments[i], i, assignments.size())) {
                return false;
            }
        }
    }

    const size_t n_stages = stages.size();

    for (const auto & stage : stages) {
        const dist_node_info * node = find_node(node_map, stage.node_id);
        if (!node) {
            continue;
        }
        shutdown_node(*node);
    }
#if !defined(_WIN32)
    usleep(300000);
#endif

    // Configure reverse: final -> ... -> entry
    for (int ri = (int) n_stages - 1; ri >= 0; --ri) {
        const auto & stage = stages[(size_t) ri];
        const dist_node_info * node = find_node(node_map, stage.node_id);
        if (!node) {
            err = "node vanished: " + stage.node_id;
            return false;
        }

        json cfg = {
            { "session_id", session_id },
            { "model_id", session.model },
            { "layer_start", stage.layer_start },
            { "layer_end", stage.layer_end },
            { "peer_bind", "0.0.0.0" },
        };

        if (const dist_model_record * record = g_registry.find(session.model)) {
            cfg["source_url"] = resolve_model_source_url(*record);
        }

        if (stage.role == DIST_ROLE_FINAL) {
            cfg["role"] = "final";
            cfg["peer_port"] = stage.peer_port;
        } else if (stage.role == DIST_ROLE_MIDDLE) {
            const auto & next = stages[(size_t) ri + 1];
            cfg["role"] = "middle";
            cfg["peer_port"] = stage.peer_port;
            cfg["next_host"] = next.host;
            cfg["next_port"] = next.peer_port;
        } else {
            const auto & next = stages[(size_t) ri + 1];
            cfg["role"] = "entry";
            cfg["ctrl_port"] = stage.ctrl_port;
            cfg["next_host"] = next.host;
            cfg["next_port"] = next.peer_port;
            cfg["next_is_final"] = (next.role == DIST_ROLE_FINAL);
        }

        if (!configure_node(*node, cfg, err)) {
            return false;
        }

#if !defined(_WIN32)
        usleep(ri == 0 ? 500000 : 200000);
#endif
    }

    session.pipeline         = stages;
    session.entry_host       = stages[0].host;
    session.entry_ctrl_port  = stages[0].ctrl_port;
    session.entry_layer_end  = stages[0].layer_end;
    session.active           = true;
    return true;
}

static json layout_json(const dist_session & session) {
    json layout = json::array();
    for (const auto & s : session.pipeline) {
        layout.push_back({
            { "node", s.node_id },
            { "start", s.layer_start },
            { "end", s.layer_end },
            { "score", s.score },
            { "role", dist_role_name(s.role) },
        });
    }
    return layout;
}

static json pipeline_json(const dist_session & session) {
    json stages = json::array();
    for (const auto & s : session.pipeline) {
        stages.push_back({
            { "node_id", s.node_id },
            { "host", s.host },
            { "role", dist_role_name(s.role) },
            { "layer_start", s.layer_start },
            { "layer_end", s.layer_end },
            { "score", s.score },
            { "ctrl_port", s.ctrl_port },
            { "peer_port", s.peer_port },
        });
    }
    return stages;
}

static bool run_generation(
        const dist_session & session,
        const std::vector<llama_token> & prompt,
        int max_new,
        std::vector<llama_token> & out_tokens,
        std::string & err) {
    if (session.pipeline.empty()) {
        err = "empty pipeline";
        return false;
    }

    const auto & entry = session.pipeline.front();
    if (entry.http_port <= 0) {
        err = "entry node missing http_port";
        return false;
    }

    json body = {
        { "prompt_tokens", json::array() },
        { "max_tokens", max_new },
        { "layer_end", session.entry_layer_end },
    };
    for (const auto t : prompt) {
        body["prompt_tokens"].push_back((int32_t) t);
    }

    httplib::Client cli(entry.host.c_str(), entry.http_port);
    cli.set_connection_timeout(10, 0);
    cli.set_read_timeout(600, 0);

    const auto res = cli.Post("/pipeline/generate", body.dump(), "application/json");
    if (!res) {
        err = "no response from entry node " + entry.node_id + " at " + entry.host + ":" +
              std::to_string(entry.http_port);
        return false;
    }
    if (res->status != 200) {
        try {
            const json j = json::parse(res->body);
            err = j.value("error", "entry node generate failed");
        } catch (...) {
            err = entry.node_id + " generate HTTP " + std::to_string(res->status);
        }
        return false;
    }

    try {
        const json j = json::parse(res->body);
        if (!j.value("ok", false)) {
            err = j.value("error", "generate failed");
            return false;
        }
        for (const auto & t : j.at("tokens")) {
            out_tokens.push_back((llama_token) t.get<int>());
        }
        return true;
    } catch (...) {
        err = "invalid generate response from entry node";
        return false;
    }
}

static void usage(const char * prog) {
    fprintf(stderr,
            "usage: %s [--model PATH] [--listen HOST:PORT] [--ctx-size N] [--models-dir DIR]\n"
            "  Layer-first mode: omit --model; register model via API, discover + manifest from HuggingFace.\n",
            prog);
}

int main(int argc, char ** argv) {
    std::string listen = "0.0.0.0:9000";

    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "--listen") == 0 && i + 1 < argc) {
            listen = argv[++i];
        } else if (strcmp(argv[i], "--model") == 0 && i + 1 < argc) {
            g_model_path = argv[++i];
        } else if (strcmp(argv[i], "--models-dir") == 0 && i + 1 < argc) {
            g_models_dir = argv[++i];
        } else if (strcmp(argv[i], "--ctx-size") == 0 && i + 1 < argc) {
            g_n_ctx = std::atoi(argv[++i]);
        } else {
            usage(argv[0]);
            return 1;
        }
    }

    if (g_models_dir.empty() && !g_model_path.empty()) {
        g_models_dir = std::filesystem::path(g_model_path).parent_path().string();
    }
    if (g_models_dir.empty()) {
        g_models_dir = default_models_dir();
    }

    g_snapshot_dir = std::filesystem::path(g_models_dir) / ".orchestrator" / "snapshots";
    g_install_journal.set_root_dir(std::filesystem::path(g_models_dir) / ".orchestrator" / "journal");

    if (registry_persistence_load(g_models_dir, g_registry)) {
        fprintf(stderr, "orchestrator: restored registry from %s/.orchestrator/registry.json\n",
                g_models_dir.c_str());
    }

    std::string bind_host;
    int port = 0;
    if (!dist_parse_host_port(listen, bind_host, port)) {
        fprintf(stderr, "orchestrator: invalid listen address\n");
        return 1;
    }

    // Initialize model catalog
    const std::string catalog_path = "state/catalog.json";
    {
        std::lock_guard<std::mutex> lock(g_mu);
        if (!g_catalog.load_catalog(catalog_path)) {
            fprintf(stderr, "orchestrator: warning - failed to load catalog from %s, using defaults\n", catalog_path.c_str());
        }
    }

    httplib::Server svr;

    svr.Get("/health", [](const httplib::Request &, httplib::Response & res) {
        res.set_content(R"({"status":"ok"})", "application/json");
    });

    svr.Post("/register", [](const httplib::Request & req, httplib::Response & res) {
        json body;
        try {
            body = json::parse(req.body);
        } catch (...) {
            res.status = 400;
            res.set_content(R"({"error":"invalid json"})", "application/json");
            return;
        }

        dist_node_info node{};
        node.node_id = body.value("node_id", "");
        node.host    = body.value("host", "127.0.0.1");
        node.http_port = body.value("port", 0);
        node.n_layer = body.value("n_layer", 0);
        node.n_embd  = body.value("n_embd", 0);
        node.memory_total_mb = body.value("memory_total_mb", static_cast<int64_t>(0));
        node.memory_free_mb  = body.value("memory_free_mb", static_cast<int64_t>(0));
        node.score = body.value("score", 1.0);
        if (body.contains("performance")) {
            const auto & perf = body["performance"];
            node.performance.score       = perf.value("score", node.score);
            node.performance.decode_tps  = perf.value("decode_tps", 0.0);
            node.performance.prefill_tps = perf.value("prefill_tps", 0.0);
            node.performance.load_ms     = perf.value("load_ms", 0.0);
            node.score = node.performance.score;
        }
        node.last_seen = dist_now_unix();
        node.online = true;

        // Task 8: byte-level memory profile.
        if (body.contains("memory")) {
            const auto & mem = body["memory"];
            node.memory.total_ram_bytes  = dist_json_u64(mem, "total_ram");
            node.memory.free_ram_bytes   = dist_json_u64(mem, "free_ram");
            node.memory.total_vram_bytes = dist_json_u64(mem, "total_vram");
            node.memory.free_vram_bytes  = dist_json_u64(mem, "free_vram");
            node.memory.has_gpu          = mem.value("has_gpu", false);

            // Mirror into the legacy capabilities block.
            node.caps.total_ram_bytes  = node.memory.total_ram_bytes;
            node.caps.free_ram_bytes   = node.memory.free_ram_bytes;
            node.caps.total_vram_bytes = node.memory.total_vram_bytes;
            node.caps.free_vram_bytes  = node.memory.free_vram_bytes;
            node.caps.has_gpu          = node.memory.has_gpu;
        }

        if (body.contains("cpu")) {
            const auto & cpu = body["cpu"];
            node.cpu.cpu_name        = cpu.value("cpu_name", "");
            node.cpu.physical_cores  = cpu.value("physical_cores", 0);
            node.cpu.logical_cores   = cpu.value("logical_cores", 0);
            node.cpu.cache_l3_bytes  = cpu.value("cache_l3", 0);
        }

        if (body.contains("system")) {
            const auto & sys = body["system"];
            node.system.os   = sys.value("os", "");
            node.system.arch = sys.value("arch", "");
        }

        if (body.contains("hardware")) {
            const auto & hw = body["hardware"];
            node.hardware.cpu_threads = hw.value("cpu_threads", 4);
            node.hardware.ram_gb      = hw.value("ram_gb", 0);
            node.hardware.gpu_name    = hw.value("gpu_name", "none");
            node.hardware.gpu_vram_gb = hw.value("gpu_vram_gb", 0);
            node.hardware.backend     = hw.value("backend", "cpu");
            node.hardware.cpu_name    = hw.value("cpu_name", node.hardware.cpu_name);

            // New fields override legacy ones if present.
            if (hw.contains("backend")) {
                node.caps.gpu_backend = hw.value("backend", node.caps.gpu_backend);
            }
            if (hw.contains("gpu_name")) {
                node.caps.gpu_name = hw.value("gpu_name", node.caps.gpu_name);
            }
        }

        if (body.contains("capabilities")) {
            const auto & caps = body["capabilities"];
            node.caps.gpu_backend   = caps.value("gpu_backend", node.caps.gpu_backend);
            node.caps.gpu_memory_mb = caps.value("gpu_memory_mb", 0);
            node.caps.cpu_threads   = caps.value("cpu_threads", 4);
            if (caps.contains("total_ram"))  node.caps.total_ram_bytes  = dist_json_u64(caps, "total_ram");
            if (caps.contains("free_ram"))   node.caps.free_ram_bytes   = dist_json_u64(caps, "free_ram");
            if (caps.contains("total_vram")) node.caps.total_vram_bytes = dist_json_u64(caps, "total_vram");
            if (caps.contains("free_vram"))  node.caps.free_vram_bytes  = dist_json_u64(caps, "free_vram");
            if (caps.contains("has_gpu"))    node.caps.has_gpu          = caps.value("has_gpu", false);
            if (caps.contains("cpu_name"))   node.caps.cpu_name = caps.value("cpu_name", "");
            if (caps.contains("os"))         node.caps.os       = caps.value("os", "");
            if (caps.contains("arch"))       node.caps.arch     = caps.value("arch", "");
        }

        if (node.node_id.empty() || node.http_port <= 0) {
            res.status = 400;
            res.set_content(R"({"error":"node_id and port required"})", "application/json");
            return;
        }

        bool cluster_changed = false;
        {
            std::lock_guard<std::mutex> lock(g_mu);
            const auto prev_it = g_nodes.find(node.node_id);
            if (prev_it == g_nodes.end()) {
                cluster_changed = true;
            } else {
                const auto & prev = prev_it->second;
                const double score_base = std::max(prev.score, 1.0);
                if (std::fabs(prev.score - node.score) / score_base >= 0.05) {
                    cluster_changed = true;
                }
                const uint64_t prev_mem = prev.memory.free_ram_bytes + prev.memory.free_vram_bytes;
                const uint64_t new_mem  = node.memory.free_ram_bytes + node.memory.free_vram_bytes;
                if (prev_mem > 0) {
                    const double mem_delta = static_cast<double>(
                            prev_mem > new_mem ? prev_mem - new_mem : new_mem - prev_mem);
                    if (mem_delta / static_cast<double>(prev_mem) >= 0.10) {
                        cluster_changed = true;
                    }
                }
            }
            g_nodes[node.node_id] = node;
        }

        // Only resync models when membership or capacity changed. Running bootstrap on
        // every heartbeat (every 5s per node) blocks the HTTP thread pool and makes
        // other nodes fail registration when a slow/unreachable node joins.
        if (cluster_changed) {
            bootstrap_all_models_from_cluster();
            trigger_cluster_optimization_async();
        }

        fprintf(stderr, "orchestrator: registered node %s at %s:%d layers=%d score=%.1f decode=%.1f prefill=%.1f "
                "ram=%dGB free_ram=%.1fGB vram=%dGB free_vram=%.1fGB has_gpu=%d\n",
                node.node_id.c_str(), node.host.c_str(), node.http_port, node.n_layer, node.score,
                node.performance.decode_tps, node.performance.prefill_tps,
                node.hardware.ram_gb,
                dist_bytes_to_gb(node.memory.free_ram_bytes),
                node.hardware.gpu_vram_gb,
                dist_bytes_to_gb(node.memory.free_vram_bytes),
                node.memory.has_gpu ? 1 : 0);

        res.set_content(json({ { "ok", true }, { "node_id", node.node_id } }).dump(), "application/json");
    });

    svr.Get("/nodes", [](const httplib::Request & req, httplib::Response & res) {
        const std::string format = req.get_param_value("format");
        const bool brief = (format == "brief");

        json nodes = json::array();
        std::lock_guard<std::mutex> lock(g_mu);
        for (const auto & kv : g_nodes) {
            const auto & n = kv.second;
            if (brief) {
                nodes.push_back({
                    { "node", n.node_id },
                    { "score", n.score },
                    { "backend", n.caps.gpu_backend },
                    { "free_ram_gb", dist_bytes_to_gb(n.memory.free_ram_bytes) },
                    { "free_vram_gb", dist_bytes_to_gb(n.memory.free_vram_bytes) },
                    { "gpu", n.hardware.gpu_name },
                });
            } else {
                nodes.push_back({
                    { "node_id", n.node_id },
                    { "host", n.host },
                    { "port", n.http_port },
                    { "n_layer", n.n_layer },
                    { "n_embd", n.n_embd },
                    { "score", n.score },
                    { "decode_tps", n.performance.decode_tps },
                    { "prefill_tps", n.performance.prefill_tps },
                    { "load_ms", n.performance.load_ms },
                    { "online", n.online },
                    { "last_seen", n.last_seen },
                    { "gpu", n.hardware.gpu_name },
                    { "ram_gb", n.hardware.ram_gb },
                    { "memory", {
                        { "total_ram", n.memory.total_ram_bytes },
                        { "free_ram", n.memory.free_ram_bytes },
                        { "total_vram", n.memory.total_vram_bytes },
                        { "free_vram", n.memory.free_vram_bytes },
                        { "has_gpu", n.memory.has_gpu },
                    }},
                    { "hardware", {
                        { "backend", n.caps.gpu_backend },
                        { "gpu_name", n.hardware.gpu_name },
                        { "cpu_name", n.cpu.cpu_name },
                        { "cpu_threads", n.hardware.cpu_threads },
                        { "ram_gb", n.hardware.ram_gb },
                        { "gpu_vram_gb", n.hardware.gpu_vram_gb },
                    }},
                    { "system", {
                        { "os", n.system.os },
                        { "arch", n.system.arch },
                    }},
                    { "performance", {
                        { "score", n.performance.score },
                        { "decode_tps", n.performance.decode_tps },
                        { "prefill_tps", n.performance.prefill_tps },
                        { "load_ms", n.performance.load_ms },
                    }},
                });
            }
        }
        res.set_content(json({ { "nodes", nodes } }).dump(), "application/json");
    });

    svr.Get("/capacity", [](const httplib::Request &, httplib::Response & res) {
        double ram_gb  = 0.0;
        double vram_gb = 0.0;
        {
            std::lock_guard<std::mutex> lock(g_mu);
            for (const auto & kv : g_nodes) {
                if (!kv.second.online) {
                    continue;
                }
                ram_gb  += dist_bytes_to_gb(kv.second.memory.total_ram_bytes);
                vram_gb += dist_bytes_to_gb(kv.second.memory.total_vram_bytes);
            }
        }
        res.set_content(json({
            { "cluster", {
                { "ram_gb", ram_gb },
                { "vram_gb", vram_gb },
            }}
        }).dump(), "application/json");
    });

    svr.Post("/session/create", [](const httplib::Request & req, httplib::Response & res) {
        json body;
        try {
            body = json::parse(req.body);
        } catch (...) {
            res.status = 400;
            res.set_content(R"({"error":"invalid json"})", "application/json");
            return;
        }

        const std::string model_id = body.value("model", "");
        const int request_ctx = body.value("n_ctx", g_n_ctx);

        if (model_id.empty()) {
            res.status = 400;
            res.set_content(json({ { "error", "model id required" } }).dump(), "application/json");
            return;
        }

        const dist_model_record * record = g_registry.find(model_id);
        if (!record) {
            res.status = 404;
            res.set_content(json({ { "error", "model not registered" } }).dump(), "application/json");
            return;
        }

        std::map<std::string, dist_node_info> node_map;
        int n_layers = 0;
        if (record->manifest.has_value() && record->manifest->n_layer > 0) {
            n_layers = static_cast<int>(record->manifest->n_layer);
        }
        {
            std::lock_guard<std::mutex> lock(g_mu);
            for (const auto & kv : g_nodes) {
                if (kv.second.online) {
                    node_map[kv.first] = kv.second;
                    if (n_layers <= 0 && kv.second.n_layer > 0) {
                        n_layers = std::max(n_layers, kv.second.n_layer);
                    }
                }
            }
        }

        if (node_map.size() < 1) {
            res.status = 503;
            res.set_content(json({ { "error", "need at least 1 registered node" } }).dump(), "application/json");
            return;
        }

        const model_memory_requirements mem = get_model_memory_for_record(*record, request_ctx);
        if (!mem.valid()) {
            res.status = 503;
            res.set_content(json({ { "error", "failed to estimate model memory" } }).dump(), "application/json");
            return;
        }

        std::vector<dist_node_info> node_vec;
        node_vec.reserve(node_map.size());
        for (const auto & kv : node_map) {
            node_vec.push_back(kv.second);
        }
        const cluster_memory_fits_result fit = dist_check_cluster_memory_fit(mem, node_vec);

        if (!fit.fits) {
            res.status = 503;
            json err = fit.to_json();
            err["error"] = "model does not fit in cluster memory";
            res.set_content(err.dump(), "application/json");
            return;
        }

        std::vector<dist_planner_node_resources> planner_nodes;
        planner_nodes.reserve(node_map.size());
        for (const auto & kv : node_map) {
            const auto & n = kv.second;
            dist_planner_node_resources r{};
            r.node_id        = n.node_id;
            r.score          = n.score;
            r.backend        = n.caps.gpu_backend;
            r.has_gpu        = n.memory.has_gpu;
            r.cpu_budget_bytes = n.memory.free_ram_bytes;
            r.gpu_budget_bytes = n.memory.free_vram_bytes;
            planner_nodes.push_back(r);
        }

        std::vector<dist_layer_assignment> assignments;
        if (record->layout.has_value() &&
                layout_has_full_coverage(record->layout->desired, n_layers)) {
            assignments = assignments_from_desired_layout(
                    record->layout->desired, node_map, n_layers);
        }

        if (assignments.empty()) {
            const dist_planner_result plan = dist_plan_layers_memory_aware(mem, planner_nodes);
            if (!plan.success) {
                res.status = 503;
                json err = fit.to_json();
                err["error"] = plan.error;
                err["planning_error"] = true;
                res.set_content(err.dump(), "application/json");
                return;
            }
            assignments = plan.assignments;
        }

        dist_print_planner_report(record->model_id, mem, node_vec, fit, assignments);

        const json planned = planned_layout_json(assignments);

        for (const auto & kv : node_map) {
            std::string herr;
            if (!check_node_health(kv.second, herr)) {
                res.status = 503;
                res.set_content(json({
                    { "error", herr },
                    { "layout", planned },
                }).dump(), "application/json");
                return;
            }
        }

        dist_session session{};
        session.session_id = make_id("sess");
        session.model      = record->model_id;
        session.model_path = resolve_tokenizer_gguf_path(*record);

        std::string err;
        if (!setup_pipeline(session.session_id, n_layers, assignments, node_map, session, err)) {
            res.status = 500;
            res.set_content(json({
                { "error", err },
                { "layout", planned },
            }).dump(), "application/json");
            return;
        }

        {
            std::lock_guard<std::mutex> lock(g_mu);
            g_sessions[session.session_id] = session;
        }

        fprintf(stderr, "orchestrator: session %s layout:", session.session_id.c_str());
        for (const auto & s : session.pipeline) {
            fprintf(stderr, " %s=[%d,%d)", s.node_id.c_str(), s.layer_start, s.layer_end);
        }
        fprintf(stderr, "\n");

        json response = {
            { "session_id", session.session_id },
            { "layout", layout_json(session) },
            { "pipeline", pipeline_json(session) },
            { "memory", {
                { "required_gb", mem.total_gb() },
                { "weights_gb", mem.weights_gb() },
                { "kv_gb", mem.kv_gb() },
                { "compute_gb", mem.compute_gb() },
                { "scratch_gb", mem.scratch_gb() },
            }},
        };
        res.set_content(response.dump(), "application/json");
    });

    svr.Post("/planner/simulate", [](const httplib::Request & req, httplib::Response & res) {
        json body;
        try {
            body = json::parse(req.body);
        } catch (...) {
            res.status = 400;
            res.set_content(R"({"error":"invalid json"})", "application/json");
            return;
        }

        const std::string model_id = body.value("model", "");
        const int request_ctx = body.value("n_ctx", g_n_ctx);

        if (model_id.empty()) {
            res.status = 400;
            res.set_content(R"({"error":"model is required"})", "application/json");
            return;
        }

        const dist_model_record * record = g_registry.find(model_id);
        if (!record) {
            res.status = 404;
            res.set_content(json({ { "error", "model not registered" } }).dump(), "application/json");
            return;
        }

        std::map<std::string, dist_node_info> node_map;
        int n_layers = 0;
        {
            std::lock_guard<std::mutex> lock(g_mu);
            for (const auto & kv : g_nodes) {
                if (kv.second.online && kv.second.n_layer > 0) {
                    node_map[kv.first] = kv.second;
                    n_layers = std::max(n_layers, kv.second.n_layer);
                }
            }
        }

        if (node_map.empty()) {
            res.status = 503;
            res.set_content(json({ { "error", "no online nodes" } }).dump(), "application/json");
            return;
        }

        const model_memory_requirements mem = get_model_memory_for_record(*record, request_ctx);
        if (!mem.valid()) {
            res.status = 503;
            res.set_content(json({ { "error", "failed to estimate model memory" } }).dump(), "application/json");
            return;
        }

        std::vector<dist_node_info> node_vec;
        node_vec.reserve(node_map.size());
        for (const auto & kv : node_map) {
            node_vec.push_back(kv.second);
        }
        const cluster_memory_fits_result fit = dist_check_cluster_memory_fit(mem, node_vec);

        json response = fit.to_json();
        response["model"] = record->model_id;
        response["n_ctx"] = request_ctx;
        response["memory"] = {
            { "weights_gb", mem.weights_gb() },
            { "kv_gb", mem.kv_gb() },
            { "compute_gb", mem.compute_gb() },
            { "scratch_gb", mem.scratch_gb() },
            { "total_gb", mem.total_gb() },
        };

        if (!fit.fits) {
            response["layout"] = json::array();
            res.status = 503;
            res.set_content(response.dump(), "application/json");
            return;
        }

        std::vector<dist_planner_node_resources> planner_nodes;
        planner_nodes.reserve(node_map.size());
        for (const auto & kv : node_map) {
            const auto & n = kv.second;
            dist_planner_node_resources r{};
            r.node_id        = n.node_id;
            r.score          = n.score;
            r.backend        = n.caps.gpu_backend;
            r.has_gpu        = n.memory.has_gpu;
            r.cpu_budget_bytes = n.memory.free_ram_bytes;
            r.gpu_budget_bytes = n.memory.free_vram_bytes;
            planner_nodes.push_back(r);
        }

        const dist_planner_result plan = dist_plan_layers_memory_aware(mem, planner_nodes);
        if (!plan.success) {
            response["fits"] = false;
            response["error"] = plan.error;
            res.status = 503;
            res.set_content(response.dump(), "application/json");
            return;
        }

        json layout = json::array();
        for (const auto & a : plan.assignments) {
            layout.push_back({
                { "node", a.node_id },
                { "layers", { a.layer_start, a.layer_end } },
                { "device_hint", a.device_hint },
            });
        }
        response["layout"] = layout;
        res.set_content(response.dump(), "application/json");
    });

    svr.Get("/planner/explain", [](const httplib::Request & req, httplib::Response & res) {
        const std::string model_id = req.get_param_value("model");
        if (model_id.empty()) {
            res.status = 400;
            res.set_content(R"({"error":"model query parameter is required"})", "application/json");
            return;
        }

        int request_ctx = g_n_ctx;
        try {
            request_ctx = std::stoi(req.get_param_value("n_ctx"));
            if (request_ctx <= 0) {
                request_ctx = g_n_ctx;
            }
        } catch (...) {
            request_ctx = g_n_ctx;
        }

        const dist_model_record * record = g_registry.find(model_id);
        if (!record) {
            res.status = 404;
            json err;
            err["error"] = "model not registered";
            res.set_content(err.dump(), "application/json");
            return;
        }

        std::map<std::string, dist_node_info> node_map;
        {
            std::lock_guard<std::mutex> lock(g_mu);
            for (const auto & kv : g_nodes) {
                if (kv.second.online && kv.second.n_layer > 0) {
                    node_map[kv.first] = kv.second;
                }
            }
        }

        json response;
        response["model"] = record->model_id;
        response["n_ctx"] = request_ctx;

        if (node_map.empty()) {
            res.status = 503;
            response["error"] = "no online nodes";
            res.set_content(response.dump(), "application/json");
            return;
        }

        const model_memory_requirements mem = get_model_memory_for_record(*record, request_ctx);
        if (!mem.valid()) {
            res.status = 503;
            response["error"] = "failed to estimate model memory";
            res.set_content(response.dump(), "application/json");
            return;
        }

        std::vector<dist_node_info> node_vec;
        node_vec.reserve(node_map.size());
        for (const auto & kv : node_map) {
            node_vec.push_back(kv.second);
        }
        const cluster_memory_fits_result fit = dist_check_cluster_memory_fit(mem, node_vec);

        response["memory"] = {
            { "weights_gb", mem.weights_gb() },
            { "kv_gb", mem.kv_gb() },
            { "compute_gb", mem.compute_gb() },
            { "scratch_gb", mem.scratch_gb() },
            { "total_gb", mem.total_gb() },
        };
        response["fits"]       = fit.fits;
        response["required_gb"] = fit.required_gb;
        response["available_gb"] = fit.available_gb;
        response["missing_gb"]   = fit.missing_gb;

        std::vector<dist_planner_node_resources> planner_nodes;
        planner_nodes.reserve(node_map.size());
        for (const auto & kv : node_map) {
            const auto & n = kv.second;
            dist_planner_node_resources r{};
            r.node_id        = n.node_id;
            r.score          = n.score;
            r.backend        = n.caps.gpu_backend;
            r.has_gpu        = n.memory.has_gpu;
            r.cpu_budget_bytes = n.memory.free_ram_bytes;
            r.gpu_budget_bytes = n.memory.free_vram_bytes;
            planner_nodes.push_back(r);
        }

        const dist_planner_result plan = dist_plan_layers_memory_aware(mem, planner_nodes);

        json nodes_json = json::array();
        json layout_json = json::array();
        std::ostringstream explanation;
        explanation.precision(1);
        explanation << std::fixed;

        explanation << "Model " << record->model_id << " requires " << mem.total_gb()
                    << " GB (weights " << mem.weights_gb()
                    << " GB, KV " << mem.kv_gb()
                    << " GB, compute " << mem.compute_gb()
                    << " GB, scratch " << mem.scratch_gb()
                    << " GB).\n";

        if (!fit.fits) {
            explanation << "The cluster is short by " << fit.missing_gb
                        << " GB of primary execution memory.\n";
        }

        for (const auto & n : planner_nodes) {
            const double ram_gb  = dist_bytes_to_gb(n.cpu_budget_bytes);
            const double vram_gb = dist_bytes_to_gb(n.gpu_budget_bytes);
            const double budget_gb = n.has_gpu ? vram_gb : ram_gb;
            const char * budget_kind = n.has_gpu ? "VRAM" : "RAM";

            int assigned_start = -1;
            int assigned_end   = -1;
            std::string device_hint = n.has_gpu ? "gpu" : "cpu";
            for (const auto & a : plan.assignments) {
                if (a.node_id == n.node_id) {
                    assigned_start = a.layer_start;
                    assigned_end   = a.layer_end;
                    device_hint    = a.device_hint;
                    break;
                }
            }

            json nj = {
                { "node_id", n.node_id },
                { "score", n.score },
                { "backend", n.backend },
                { "free_ram_gb", ram_gb },
                { "free_vram_gb", vram_gb },
                { "primary_budget_gb", budget_gb },
                { "primary_budget_kind", budget_kind },
                { "assigned_layers_start", assigned_start },
                { "assigned_layers_end", assigned_end },
                { "assigned_layer_count", assigned_end >= 0 ? assigned_end - assigned_start : 0 },
                { "assigned_device", device_hint },
            };
            nodes_json.push_back(nj);

            explanation << "Node " << n.node_id
                        << " (" << n.backend << ") has " << budget_gb
                        << " GB " << budget_kind << " budget";
            if (plan.success && assigned_end > assigned_start) {
                explanation << " and runs layers " << assigned_start
                            << "-" << assigned_end << " on " << device_hint;
                layout_json.push_back({
                    { "node", n.node_id },
                    { "layers", { assigned_start, assigned_end } },
                    { "device_hint", device_hint },
                });
            } else if (!plan.success) {
                explanation << " but could not be assigned any layer within budget";
            } else {
                explanation << " and was not used";
            }
            explanation << ".\n";
        }

        response["nodes"] = nodes_json;
        response["layout"] = layout_json;
        response["explanation"] = explanation.str();

        if (!plan.success) {
            res.status = 503;
            response["error"] = plan.error;
        }
        res.set_content(response.dump(), "application/json");
    });

    svr.Post("/session/generate", [](const httplib::Request & req, httplib::Response & res) {
        json body;
        try {
            body = json::parse(req.body);
        } catch (...) {
            res.status = 400;
            res.set_content(R"({"error":"invalid json"})", "application/json");
            return;
        }

        const std::string session_id = body.value("session_id", "");
        const std::string prompt     = body.value("prompt", SPLIT_GEN_PROMPT);
        const int max_tokens         = body.value("max_tokens", DIST_MAX_NEW_TOKENS);

        dist_session session{};
        {
            std::lock_guard<std::mutex> lock(g_mu);
            const auto it = g_sessions.find(session_id);
            if (it == g_sessions.end()) {
                res.status = 404;
                res.set_content(R"({"error":"session not found"})", "application/json");
                return;
            }
            session = it->second;
        }

        if (!session.active) {
            res.status = 503;
            res.set_content(R"({"error":"session inactive"})", "application/json");
            return;
        }

        ggml_backend_load_all();
        const dist_model_record * record = g_registry.find(session.model);
        std::string tokenizer_path = session.model_path;
        auto tokenizer_usable = [&](const std::string & path) -> bool {
            if (path.empty()) {
                return false;
            }
            std::error_code ec;
            if (!std::filesystem::exists(path, ec)) {
                return false;
            }
            if (record != nullptr && record->manifest.has_value()) {
                const uint64_t meta_size = record->manifest->tensor_data_offset;
                const auto fsize = std::filesystem::file_size(path, ec);
                if (!ec && meta_size > 0 && fsize <= meta_size) {
                    return false;
                }
            }
            return true;
        };
        if (!tokenizer_usable(tokenizer_path) && record != nullptr) {
            tokenizer_path = resolve_tokenizer_gguf_path(*record);
        }
        if (!tokenizer_usable(tokenizer_path)) {
            res.status = 503;
            res.set_content(json({
                { "error", "no tokenizer GGUF available; build manifest and cache metadata first" },
            }).dump(), "application/json");
            return;
        }

        llama_model * model = llama_model_load_from_file(tokenizer_path.c_str(), llama_model_default_params());
        if (!model && record != nullptr) {
            std::error_code ec;
            std::filesystem::remove(tokenizer_path, ec);
            tokenizer_path = resolve_tokenizer_gguf_path(*record);
            if (tokenizer_usable(tokenizer_path)) {
                model = llama_model_load_from_file(tokenizer_path.c_str(), llama_model_default_params());
            }
        }
        if (!model) {
            res.status = 500;
            res.set_content(R"({"error":"failed to load model for tokenization"})", "application/json");
            return;
        }

        const llama_vocab * vocab = llama_model_get_vocab(model);
        const std::vector<llama_token> prompt_tokens = split_gen_tokenize(vocab, prompt);

        std::vector<llama_token> tokens;
        std::string err;
        const bool ok = run_generation(session, prompt_tokens, max_tokens, tokens, err);

        json out_tokens = json::array();
        std::string text;
        for (const auto t : tokens) {
            out_tokens.push_back((int) t);
            text += split_gen_token_text(vocab, t);
        }

        llama_model_free(model);

        if (!ok) {
            res.status = 503;
            res.set_content(json({
                { "error", err },
                { "tokens", out_tokens },
                { "text", text },
            }).dump(), "application/json");
            return;
        }

        res.set_content(json({
            { "session_id", session_id },
            { "tokens", out_tokens },
            { "text", text },
            { "count", tokens.size() },
        }).dump(), "application/json");
    });

    svr.Delete(R"(/session/([^/]+))", [](const httplib::Request & req, httplib::Response & res) {
        const std::string session_id = req.matches[1];
        bool removed = false;
        {
            std::lock_guard<std::mutex> lock(g_mu);
            removed = g_sessions.erase(session_id) > 0;
        }
        if (!removed) {
            res.status = 404;
            res.set_content(json({ { "error", "session not found" } }).dump(), "application/json");
            return;
        }
        res.set_content(json({ { "session_id", session_id }, { "destroyed", true } }).dump(), "application/json");
    });

    svr.Post("/session/destroy", [](const httplib::Request & req, httplib::Response & res) {
        json body;
        try {
            body = json::parse(req.body);
        } catch (...) {
            res.status = 400;
            res.set_content(R"({"error":"invalid json"})", "application/json");
            return;
        }
        const std::string session_id = body.value("session_id", "");
        if (session_id.empty()) {
            res.status = 400;
            res.set_content(json({ { "error", "session_id required" } }).dump(), "application/json");
            return;
        }
        bool removed = false;
        {
            std::lock_guard<std::mutex> lock(g_mu);
            removed = g_sessions.erase(session_id) > 0;
        }
        if (!removed) {
            res.status = 404;
            res.set_content(json({ { "error", "session not found" } }).dump(), "application/json");
            return;
        }
        res.set_content(json({ { "session_id", session_id }, { "destroyed", true } }).dump(), "application/json");
    });

    // Model management API

    // GET /catalog - List available models in catalog
    svr.Get("/catalog", [](const httplib::Request &, httplib::Response & res) {
        std::vector<model_info> models;
        {
            std::lock_guard<std::mutex> lock(g_mu);
            models = g_catalog.get_models();
        }

        json catalog_json = json::array();
        for (const auto & model : models) {
            catalog_json.push_back({
                { "id", model.id },
                { "display_name", model.display_name },
                { "size_gb", model.size_gb },
                { "n_layers", model.n_layers },
                { "n_embd", model.n_embd }
            });
        }

        res.set_content(catalog_json.dump(), "application/json");
    });

    // -----------------------------------------------------------------------
    // Cluster Model Registry (Task 9.1)
    // -----------------------------------------------------------------------

    // POST /models/register - Register (or update) a model in the cluster.
    svr.Post("/models/register", [](const httplib::Request & req, httplib::Response & res) {
        json body;
        try {
            body = json::parse(req.body);
        } catch (...) {
            res.status = 400;
            res.set_content(R"({"error":"invalid json"})", "application/json");
            return;
        }

        if (!body.contains("model_id") || body.value("model_id", "").empty()) {
            res.status = 400;
            res.set_content(json({ { "error", "model_id is required" } }).dump(), "application/json");
            return;
        }

        dist_model_record incoming = dist_model_record_from_json(body);
        const std::string model_id = incoming.model_id;

        if (dist_model_record * existing = g_registry.find(model_id)) {
            dist_model_record merged = *existing;
            if (!incoming.display_name.empty()) merged.display_name = incoming.display_name;
            if (!incoming.source.empty())       merged.source       = incoming.source;
            if (!incoming.repository.empty())   merged.repository   = incoming.repository;
            if (!incoming.filename.empty())     merged.filename     = incoming.filename;
            if (!incoming.revision.empty())     merged.revision     = incoming.revision;
            g_registry.add_or_update(merged);
            persist_registry();
            res.set_content(merged.to_json().dump(), "application/json");
            return;
        }

        incoming.status = dist_model_status::discovered;
        g_registry.add_or_update(incoming);
        persist_registry();

        res.set_content(incoming.to_json().dump(), "application/json");
    });

    // GET /models - List all registered models.
    svr.Get("/models", [](const httplib::Request &, httplib::Response & res) {
        const auto records = g_registry.list();
        json out = json::array();
        for (const auto & r : records) {
            out.push_back(r.to_json());
        }
        res.set_content(out.dump(), "application/json");
    });

    // GET /models/installed - List installed models across cluster (aggregated from nodes)
    svr.Get("/models/installed", [](const httplib::Request &, httplib::Response & res) {
        std::vector<dist_node_info> nodes;
        {
            std::lock_guard<std::mutex> lock(g_mu);
            for (const auto & kv : g_nodes) {
                if (kv.second.online) {
                    nodes.push_back(kv.second);
                }
            }
        }

        // model_id -> aggregated info
        struct agg_model {
            int total_nodes = 0;
            int ready_nodes = 0;
            json nodes = json::array();
        };
        std::map<std::string, agg_model> models;

        for (const auto & node : nodes) {
            httplib::Client client(node.host.c_str(), node.http_port);
            client.set_connection_timeout(3, 0);
            client.set_read_timeout(10, 0);

            const auto result = client.Get("/models/local");
            if (!result || result->status != 200) {
                continue;
            }

            json local;
            try {
                local = json::parse(result->body);
            } catch (...) {
                continue;
            }
            if (!local.is_array()) {
                continue;
            }

            for (const auto & m : local) {
                const std::string id = m.value("model_id", "");
                if (id.empty()) {
                    continue;
                }
                const bool ready = m.value("ready", false);
                auto & a = models[id];
                a.total_nodes += 1;
                if (ready) {
                    a.ready_nodes += 1;
                }
                a.nodes.push_back({
                    { "node_id", node.node_id },
                    { "ready", ready },
                    { "status", m.value("status", "unknown") },
                    { "local_path", m.value("local_path", "") },
                });
            }
        }

        json out = json::array();
        for (const auto & [id, a] : models) {
            std::string status = "unknown";
            if (a.ready_nodes > 0 && a.ready_nodes == a.total_nodes) {
                status = "ready";
            } else if (a.ready_nodes > 0) {
                status = "partial";
            } else if (a.total_nodes > 0) {
                status = "downloading";
            }
            out.push_back({
                { "id", id },
                { "status", status },
                { "ready_nodes", a.ready_nodes },
                { "total_nodes", a.total_nodes },
                { "nodes", a.nodes },
            });
        }

        res.set_content(out.dump(), "application/json");
    });

    // POST /models/install - Start installing a model across the cluster.
    svr.Post("/models/install", [](const httplib::Request & req, httplib::Response & res) {
        json body;
        try {
            body = json::parse(req.body);
        } catch (...) {
            res.status = 400;
            res.set_content(R"({"error":"invalid json"})", "application/json");
            return;
        }

        const std::string model_id = body.value("model", "");
        if (model_id.empty()) {
            res.status = 400;
            res.set_content(R"({"error":"model id required"})", "application/json");
            return;
        }

        std::string job_id;
        model_info model;
        std::vector<dist_node_info> online_nodes;
        {
            std::lock_guard<std::mutex> lock(g_mu);

            const dist_model_record * reg = g_registry.find(model_id);
            if (reg) {
                model.id           = reg->model_id;
                model.display_name = reg->display_name;
                model.source.repo  = reg->repository;
                model.source.file  = reg->filename;
            } else {
                const auto * found = g_catalog.find_model(model_id);
                if (!found) {
                    res.status = 404;
                    res.set_content(R"({"error":"model not found in catalog"})", "application/json");
                    return;
                }
                model = *found;
            }

            job_id = g_catalog.create_install_job(model_id);
            g_catalog.update_job_status(job_id, install_status::downloading, "", 0.0);
            for (const auto & [_, node] : g_nodes) {
                if (node.online) {
                    online_nodes.push_back(node);
                    g_install_node_results[job_id][node.node_id] = {
                        { "status", "pending" },
                        { "host", node.host },
                        { "port", node.http_port }
                    };
                }
            }
        }

        std::thread(coordinate_model_install, job_id, model_id, model, online_nodes).detach();

        res.set_content(json({
            { "job_id", job_id },
            { "model", model_id },
            { "status", "started" }
        }).dump(), "application/json");
    });

    // GET /models/install/{job_id} - Check installation status
    svr.Get(R"(/models/install/(.+))", [](const httplib::Request & req, httplib::Response & res) {
        install_job job;
        json node_results = json::object();
        {
            std::lock_guard<std::mutex> lock(g_mu);
            auto * found = g_catalog.get_install_job(req.matches[1]);
            if (!found) {
                res.status = 404;
                res.set_content(R"({"error":"job not found"})", "application/json");
                return;
            }
            job = *found;
            const auto node_it = g_install_node_results.find(job.job_id);
            if (node_it != g_install_node_results.end()) {
                node_results = node_it->second;
            }
        }

        std::string status_str;
        switch (job.status) {
            case install_status::unknown:     status_str = "unknown"; break;
            case install_status::downloading: status_str = "downloading"; break;
            case install_status::ready:       status_str = "ready"; break;
            case install_status::error:       status_str = "error"; break;
        }

        json response = {
            { "job_id", job.job_id },
            { "model", job.model_id },
            { "status", status_str },
            { "progress", job.progress },
            { "nodes", node_results }
        };

        if (!job.error_msg.empty()) {
            response["error"] = job.error_msg;
        }

        res.set_content(response.dump(), "application/json");
    });

    // POST /models/{model_id}/discover - Trigger remote discovery for a model.
    svr.Post(R"(/models/([^/]+)/discover)", [](const httplib::Request & req, httplib::Response & res) {
        const std::string model_id = req.matches[1];

        dist_model_record * record = g_registry.find(model_id);
        if (!record) {
            res.status = 404;
            res.set_content(json({ { "error", "model not registered" } }).dump(), "application/json");
            return;
        }

        const std::string provider_name = record->source.empty() ? "huggingface" : record->source;
        auto provider = create_model_provider(provider_name);
        if (!provider) {
            res.status = 400;
            res.set_content(json({ { "error", "unknown provider: " + provider_name } }).dump(), "application/json");
            return;
        }

        const auto result = provider->discover(*record);
        if (!result.success) {
            res.status = 502;
            res.set_content(json({ { "error", result.error } }).dump(), "application/json");
            return;
        }

        if (!g_registry.apply_discovery(model_id, result, record)) {
            res.status = 404;
            res.set_content(json({ { "error", "model not registered" } }).dump(), "application/json");
            return;
        }

        res.set_content(json({
            { "status", "ok" },
            { "provider", result.provider },
            { "files", static_cast<int>(result.files.size()) },
            { "revision", result.revision }
        }).dump(), "application/json");
    });

    // POST /models/{model_id}/manifest - Build GGUF manifest (metadata only).
    svr.Post(R"(/models/([^/]+)/manifest)", [](const httplib::Request & req, httplib::Response & res) {
        const std::string model_id = req.matches[1];

        dist_model_record * record = g_registry.find(model_id);
        if (!record) {
            res.status = 404;
            res.set_content(json({ { "error", "model not registered" } }).dump(), "application/json");
            return;
        }

        const auto build = build_manifest_for_record(*record, g_models_dir, g_model_path);
        if (!build.success) {
            res.status = 502;
            res.set_content(json({ { "error", build.error } }).dump(), "application/json");
            return;
        }

        if (!g_registry.apply_manifest(model_id, build.manifest, record)) {
            res.status = 404;
            res.set_content(json({ { "error", "model not registered" } }).dump(), "application/json");
            return;
        }

        cache_manifest_metadata(model_id, build.manifest);
        persist_registry();

        res.set_content(json({
            { "status", "ok" },
            { "architecture", build.manifest.architecture },
            { "n_layer", build.manifest.n_layer },
            { "n_ctx", build.manifest.n_ctx },
            { "tensors", static_cast<int>(build.manifest.tensors.size()) },
            { "layers", static_cast<int>(build.manifest.layers.size()) },
            { "metadata_bytes_read", build.bytes_read },
        }).dump(), "application/json");

        if (!record->layout.has_value() ||
                !record->coverage.has_value() ||
                record->coverage->state != coverage_state::ready) {
            trigger_cluster_optimization_async();
        }
    });

    // GET /models/{model_id}/manifest - Return the stored manifest.
    svr.Get(R"(/models/([^/]+)/manifest)", [](const httplib::Request & req, httplib::Response & res) {
        const std::string model_id = req.matches[1];
        const auto * record = g_registry.find(model_id);
        if (!record) {
            res.status = 404;
            res.set_content(json({ { "error", "model not registered" } }).dump(), "application/json");
            return;
        }
        if (record->status != dist_model_status::manifest_ready || !record->manifest.has_value()) {
            res.status = 404;
            res.set_content(json({ { "error", "manifest not ready" } }).dump(), "application/json");
            return;
        }
        res.set_content(record->manifest->to_json().dump(), "application/json");
    });

    // POST /models/{model_id}/layout - Return stored layout or build if missing.
    // Pass {"force": true} to recompute layout from the planner.
    svr.Post(R"(/models/([^/]+)/layout)", [](const httplib::Request & req, httplib::Response & res) {
        const std::string model_id = req.matches[1];

        dist_model_record * record = g_registry.find(model_id);
        if (!record) {
            res.status = 404;
            res.set_content(json({ { "error", "model not registered" } }).dump(), "application/json");
            return;
        }
        if (record->status != dist_model_status::manifest_ready || !record->manifest.has_value()) {
            res.status = 404;
            res.set_content(json({ { "error", "manifest not ready" } }).dump(), "application/json");
            return;
        }

        bool force = false;
        int request_ctx = g_n_ctx;
        try {
            if (!req.body.empty()) {
                const json body = json::parse(req.body);
                force = body.value("force", false);
                if (body.contains("n_ctx")) {
                    request_ctx = body.value("n_ctx", g_n_ctx);
                }
            }
        } catch (...) {}

        sync_model_state_from_cluster(model_id, true);
        record = g_registry.find(model_id);
        if (!record) {
            res.status = 404;
            res.set_content(json({ { "error", "model not registered" } }).dump(), "application/json");
            return;
        }

        if (!force && record->layout.has_value() &&
                !record->layout->desired.placements.empty()) {
            const auto & layout = record->layout->desired;
            res.set_content(json({
                { "status", "ok" },
                { "cached", true },
                { "model", model_id },
                { "fits_cluster", layout.fits_cluster },
                { "placements", static_cast<int>(layout.placements.size()) },
                { "coverage", record->coverage.has_value()
                        ? coverage_state_to_string(record->coverage->state)
                        : "unknown" },
                { "total_weight_bytes", layout.total_weight_bytes },
                { "total_required_memory", layout.total_required_memory },
            }).dump(), "application/json");
            return;
        }

        std::vector<layout_node_input> nodes;
        {
            std::lock_guard<std::mutex> lock(g_mu);
            for (const auto & kv : g_nodes) {
                if (kv.second.online) {
                    nodes.push_back(layout_node_from_dist(kv.second));
                }
            }
        }

        desired_model_layout previous;
        if (record->layout.has_value()) {
            previous = record->layout->desired;
        }

        const auto built = build_desired_layout(model_id, *record->manifest, nodes, request_ctx);
        if (!built.success) {
            res.status = 502;
            res.set_content(json({
                { "error", built.error },
                { "fits_cluster", built.layout.fits_cluster },
                { "warnings", built.layout.warnings },
            }).dump(), "application/json");
            return;
        }

        if (!g_registry.apply_layout(model_id, built.layout, record)) {
            res.status = 404;
            res.set_content(json({ { "error", "model not registered" } }).dump(), "application/json");
            return;
        }

        sync_model_state_from_cluster(model_id, false);
        persist_registry();

        res.set_content(json({
            { "status", "ok" },
            { "cached", false },
            { "model", model_id },
            { "fits_cluster", built.layout.fits_cluster },
            { "placements", static_cast<int>(built.layout.placements.size()) },
            { "total_weight_bytes", built.layout.total_weight_bytes },
            { "total_required_memory", built.layout.total_required_memory },
        }).dump(), "application/json");

        if (!layouts_placement_equal(previous, built.layout)) {
            trigger_cluster_optimization_async();
        }
    });

    // GET /models/{model_id}/layout - Return stored desired layout.
    svr.Get(R"(/models/([^/]+)/layout)", [](const httplib::Request & req, httplib::Response & res) {
        const std::string model_id = req.matches[1];
        const auto * record = g_registry.find(model_id);
        if (!record) {
            res.status = 404;
            res.set_content(json({ { "error", "model not registered" } }).dump(), "application/json");
            return;
        }
        if (!record->layout.has_value()) {
            res.status = 404;
            res.set_content(json({ { "error", "layout not ready" } }).dump(), "application/json");
            return;
        }
        res.set_content(record->layout->to_json().dump(), "application/json");
    });

    // POST /models/{model_id}/optimize - Run cluster optimizer (Task 9.8).
    svr.Post(R"(/models/([^/]+)/optimize)", [](const httplib::Request & req, httplib::Response & res) {
        const std::string model_id = req.matches[1];

        dist_model_record * record = g_registry.find(model_id);
        if (!record) {
            res.status = 404;
            res.set_content(json({ { "error", "model not registered" } }).dump(), "application/json");
            return;
        }
        if (!record->manifest.has_value()) {
            res.status = 404;
            res.set_content(json({ { "error", "manifest not ready" } }).dump(), "application/json");
            return;
        }

        optimizer_policy policy{};
        try {
            if (!req.body.empty()) {
                const json body = json::parse(req.body);
                if (body.contains("min_decode_improvement_percent")) {
                    policy.min_decode_improvement_percent = body.value("min_decode_improvement_percent", 5.0);
                }
                if (body.contains("min_prefill_improvement_percent")) {
                    policy.min_prefill_improvement_percent = body.value("min_prefill_improvement_percent", 5.0);
                }
                if (body.contains("max_rebalance_cost_bytes")) {
                    policy.max_rebalance_cost_bytes = body.value("max_rebalance_cost_bytes", UINT64_MAX);
                }
                if (body.contains("allow_storage_only_nodes")) {
                    policy.allow_storage_only_nodes = body.value("allow_storage_only_nodes", true);
                }
            }
        } catch (...) {
            res.status = 400;
            res.set_content(R"({"error":"invalid json"})", "application/json");
            return;
        }

        const optimization_result result = optimize_registered_model(model_id, &policy);
        if (result.decision == optimizer_decision::rebalance) {
            std::thread([model_id]() {
                coordinate_rebalance_pipeline(model_id);
            }).detach();
        }

        res.set_content(result.to_json().dump(), "application/json");
    });

    // GET /models/{model_id}/optimization - Last optimizer result (Task 9.8).
    svr.Get(R"(/models/([^/]+)/optimization)", [](const httplib::Request & req, httplib::Response & res) {
        const std::string model_id = req.matches[1];
        const auto * record = g_registry.find(model_id);
        if (!record) {
            res.status = 404;
            res.set_content(json({ { "error", "model not registered" } }).dump(), "application/json");
            return;
        }
        if (!record->optimization.has_value()) {
            res.status = 404;
            res.set_content(json({ { "error", "optimization not run yet" } }).dump(), "application/json");
            return;
        }
        res.set_content(record->optimization->to_json().dump(), "application/json");
    });

    // POST /models/{model_id}/coverage/refresh - Poll nodes and recompute coverage.
    svr.Post(R"(/models/([^/]+)/coverage/refresh)", [](const httplib::Request & req, httplib::Response & res) {
        const std::string model_id = req.matches[1];

        dist_model_record * record = g_registry.find(model_id);
        if (!record) {
            res.status = 404;
            res.set_content(json({ { "error", "model not registered" } }).dump(), "application/json");
            return;
        }
        if (!record->layout.has_value()) {
            res.status = 404;
            res.set_content(json({ { "error", "layout not ready" } }).dump(), "application/json");
            return;
        }

        actual_model_layout actual;
        std::set<std::string> online_nodes;
        poll_installed_layers_from_nodes(model_id, actual, online_nodes);

        if (!g_registry.apply_actual(model_id, actual, record)) {
            res.status = 404;
            res.set_content(json({ { "error", "model not registered" } }).dump(), "application/json");
            return;
        }
        if (!g_registry.refresh_coverage(model_id, online_nodes, record)) {
            res.status = 500;
            res.set_content(json({ { "error", "coverage refresh failed" } }).dump(), "application/json");
            return;
        }

        res.set_content(json({
            { "status", "ok" },
            { "coverage", record->coverage->to_json() },
        }).dump(), "application/json");
    });

    // GET /models/{model_id}/coverage - Return stored coverage report.
    svr.Get(R"(/models/([^/]+)/coverage)", [](const httplib::Request & req, httplib::Response & res) {
        const std::string model_id = req.matches[1];
        const auto * record = g_registry.find(model_id);
        if (!record) {
            res.status = 404;
            res.set_content(json({ { "error", "model not registered" } }).dump(), "application/json");
            return;
        }
        if (!record->coverage.has_value()) {
            res.status = 404;
            res.set_content(json({ { "error", "coverage not ready" } }).dump(), "application/json");
            return;
        }
        res.set_content(record->coverage->to_json().dump(), "application/json");
    });

    // POST /models/{model_id}/reconcile - Refresh actual state and return differences.
    svr.Post(R"(/models/([^/]+)/reconcile)", [](const httplib::Request & req, httplib::Response & res) {
        const std::string model_id = req.matches[1];

        dist_model_record * record = g_registry.find(model_id);
        if (!record) {
            res.status = 404;
            res.set_content(json({ { "error", "model not registered" } }).dump(), "application/json");
            return;
        }
        if (!record->layout.has_value()) {
            res.status = 404;
            res.set_content(json({ { "error", "layout not ready" } }).dump(), "application/json");
            return;
        }

        actual_model_layout actual;
        std::set<std::string> online_nodes;
        poll_installed_layers_from_nodes(model_id, actual, online_nodes);

        if (!g_registry.apply_actual(model_id, actual, record)) {
            res.status = 404;
            res.set_content(json({ { "error", "model not registered" } }).dump(), "application/json");
            return;
        }
        g_registry.refresh_coverage(model_id, online_nodes, record);

        const reconciliation_result result = reconcile_layers(
                record->layout->desired, actual, online_nodes);
        coverage_report coverage = compute_coverage(
                record->layout->desired, actual, online_nodes);
        if (record->manifest.has_value()) {
            const semantic_runtime_descriptor rt =
                    build_semantic_runtime_descriptor(*record->manifest);
            coverage = compute_runtime_coverage(
                    rt, record->layout->desired, actual, online_nodes).layer_coverage;
        }
        g_registry.apply_coverage(model_id, coverage, record);

        res.set_content(result.to_json().dump(), "application/json");
    });

    // POST /models/{model_id}/install-plan - Build install plan from registry state.
    svr.Post(R"(/models/([^/]+)/install-plan)", [](const httplib::Request & req, httplib::Response & res) {
        const std::string model_id = req.matches[1];

        dist_model_record * record = g_registry.find(model_id);
        if (!record) {
            res.status = 404;
            res.set_content(json({ { "error", "model not registered" } }).dump(), "application/json");
            return;
        }
        if (!record->manifest.has_value()) {
            res.status = 404;
            res.set_content(json({ { "error", "manifest not ready" } }).dump(), "application/json");
            return;
        }

        const desired_model_layout * target = planning_target_layout(*record);
        if (!target && !record->layout.has_value()) {
            res.status = 404;
            res.set_content(json({ { "error", "layout not ready" } }).dump(), "application/json");
            return;
        }

        if (!build_and_store_install_plan(model_id, record)) {
            res.status = 502;
            res.set_content(json({ { "error", "install plan build failed" } }).dump(), "application/json");
            return;
        }

        record = g_registry.find(model_id);
        if (!record || !record->stored_install_plan.has_value()) {
            res.status = 500;
            res.set_content(json({ { "error", "install plan not stored" } }).dump(), "application/json");
            return;
        }

        res.set_content(json({
            { "status", "ok" },
            { "model", model_id },
            { "operation_count", record->stored_install_plan->operation_count },
            { "total_download_bytes", record->stored_install_plan->total_download_bytes },
            { "install_plan", record->stored_install_plan->to_json() },
        }).dump(), "application/json");
    });

    // GET /models/{model_id}/install-plan - Return stored install plan.
    svr.Get(R"(/models/([^/]+)/install-plan)", [](const httplib::Request & req, httplib::Response & res) {
        const std::string model_id = req.matches[1];
        const auto * record = g_registry.find(model_id);
        if (!record) {
            res.status = 404;
            res.set_content(json({ { "error", "model not registered" } }).dump(), "application/json");
            return;
        }
        if (!record->stored_install_plan.has_value()) {
            res.status = 404;
            res.set_content(json({ { "error", "install plan not ready" } }).dump(), "application/json");
            return;
        }
        res.set_content(record->stored_install_plan->to_json().dump(), "application/json");
    });

    // POST /models/{model_id}/install/execute - Execute stored install plan via nodes.
    svr.Post(R"(/models/([^/]+)/install/execute)", [](const httplib::Request & req, httplib::Response & res) {
        const std::string model_id = req.matches[1];

        dist_model_record * record = g_registry.find(model_id);
        if (!record) {
            res.status = 404;
            res.set_content(json({ { "error", "model not registered" } }).dump(), "application/json");
            return;
        }
        if (!record->stored_install_plan.has_value()) {
            res.status = 404;
            res.set_content(json({ { "error", "install plan not ready" } }).dump(), "application/json");
            return;
        }
        if (!record->manifest.has_value()) {
            res.status = 404;
            res.set_content(json({ { "error", "manifest not ready" } }).dump(), "application/json");
            return;
        }

        std::map<std::string, dist_node_info> nodes_copy;
        {
            std::lock_guard<std::mutex> lock(g_mu);
            for (const auto & kv : g_nodes) {
                if (kv.second.online) {
                    nodes_copy[kv.first] = kv.second;
                }
            }
        }

        const std::string job_id = make_id("job");
        {
            std::lock_guard<std::mutex> lock(g_mu);
            cluster_sync_job job;
            job.job_id   = job_id;
            job.model_id = model_id;
            job.state    = "queued";
            g_sync_jobs[job_id] = std::move(job);
        }

        std::thread(
                coordinate_install_plan_execute,
                job_id,
                model_id,
                *record->stored_install_plan,
                *record->manifest,
                nodes_copy).detach();

        res.set_content(json({
            { "job_id", job_id },
            { "model_id", model_id },
            { "status", "started" },
            { "operation_count", record->stored_install_plan->operation_count },
        }).dump(), "application/json");
    });

    // POST /models/{model_id}/verify - Run GGUF materialization verification pipeline.
    svr.Post(R"(/models/([^/]+)/verify)", [](const httplib::Request & req, httplib::Response & res) {
        const std::string model_id = req.matches[1];
        const dist_model_record * record = g_registry.find(model_id);
        if (!record || !record->manifest.has_value()) {
            res.status = 404;
            res.set_content(json({ { "error", "model not found or manifest missing" } }).dump(), "application/json");
            return;
        }

        const std::string original = resolve_model_path(*record);
        if (original.empty()) {
            res.status = 503;
            res.set_content(json({
                { "error", "original GGUF not available locally; set --models-dir or register with local file" },
            }).dump(), "application/json");
            return;
        }

        layer_store store(g_models_dir.empty() ? std::filesystem::path(original).parent_path().string() : g_models_dir,
                model_id);
        const std::string work_dir = (store.model_root() / "verify").string();
        const verification_report report = run_verification_pipeline(
                model_id, original, store, *record->manifest, work_dir);

        {
            std::lock_guard<std::mutex> lock(g_mu);
            g_verify_reports[model_id] = report.to_json();
        }

        res.set_content(report.summary_json().dump(), "application/json");
        if (!report.passed) {
            res.status = 422;
        }
    });

    // GET /models/{model_id}/verify - Return last verification report.
    svr.Get(R"(/models/([^/]+)/verify)", [](const httplib::Request & req, httplib::Response & res) {
        const std::string model_id = req.matches[1];
        std::lock_guard<std::mutex> lock(g_mu);
        const auto it = g_verify_reports.find(model_id);
        if (it == g_verify_reports.end()) {
            res.status = 404;
            res.set_content(json({ { "error", "no verification report; POST /verify first" } }).dump(),
                    "application/json");
            return;
        }
        res.set_content(it->second.dump(), "application/json");
    });

    // GET /jobs/{job_id} - Cluster synchronization job progress.
    svr.Get(R"(/jobs/(.+))", [](const httplib::Request & req, httplib::Response & res) {
        const std::string job_id = req.matches[1];
        std::lock_guard<std::mutex> lock(g_mu);
        const auto it = g_sync_jobs.find(job_id);
        if (it == g_sync_jobs.end()) {
            res.status = 404;
            res.set_content(json({ { "error", "job not found" } }).dump(), "application/json");
            return;
        }

        json nodes = json::object();
        for (const auto & kv : it->second.node_status) {
            nodes[kv.first] = kv.second;
        }

        res.set_content(json({
            { "job_id", it->second.job_id },
            { "model_id", it->second.model_id },
            { "state", it->second.state },
            { "nodes", nodes },
            { "error", it->second.error },
        }).dump(), "application/json");
    });

    // GET /models/{model_id} - Return one registered model.
    svr.Get(R"(/models/([^/]+))", [](const httplib::Request & req, httplib::Response & res) {
        const std::string model_id = req.matches[1];
        const auto * record = g_registry.find(model_id);
        if (!record) {
            res.status = 404;
            res.set_content(json({ { "error", "model not registered" } }).dump(), "application/json");
            return;
        }
        res.set_content(record->to_json().dump(), "application/json");
    });

    // POST /models/{model_id}/consistency - Full cluster consistency check.
    svr.Post(R"(/models/([^/]+)/consistency)", [](const httplib::Request & req, httplib::Response & res) {
        const std::string model_id = req.matches[1];
        dist_model_record * record = g_registry.find(model_id);
        if (!record || !record->layout.has_value() || !record->manifest.has_value()) {
            res.status = 404;
            res.set_content(json({ { "error", "model not ready" } }).dump(), "application/json");
            return;
        }

        actual_model_layout actual;
        std::set<std::string> online_nodes;
        poll_installed_layers_from_nodes(model_id, actual, online_nodes);
        g_registry.apply_actual(model_id, actual, record);
        g_registry.refresh_coverage(model_id, online_nodes, record);
        record = g_registry.find(model_id);
        if (!record) {
            res.status = 500;
            res.set_content(json({ { "error", "registry lost model" } }).dump(), "application/json");
            return;
        }

        build_and_store_install_plan(model_id, record);
        record = g_registry.find(model_id);

        const consistency_check_result check = check_cluster_consistency(
                *record,
                actual,
                online_nodes,
                resolve_model_source_url(*record));

        persist_registry();

        res.set_content(check.to_json().dump(), "application/json");
    });

    // POST /models/{model_id}/reset - Clear node layer stores and registry install state.
    svr.Post(R"(/models/([^/]+)/reset)", [](const httplib::Request & req, httplib::Response & res) {
        const std::string model_id = req.matches[1];
        dist_model_record * record = g_registry.find(model_id);
        if (!record) {
            res.status = 404;
            res.set_content(json({ { "error", "model not registered" } }).dump(), "application/json");
            return;
        }

        json body;
        try {
            body = req.body.empty() ? json::object() : json::parse(req.body);
        } catch (...) {
            body = json::object();
        }
        const bool keep_manifest = body.value("keep_manifest", true);

        json node_results = json::object();
        std::map<std::string, dist_node_info> nodes_copy;
        {
            std::lock_guard<std::mutex> lock(g_mu);
            for (const auto & kv : g_nodes) {
                if (kv.second.online) {
                    nodes_copy[kv.first] = kv.second;
                }
            }
        }

        for (const auto & kv : nodes_copy) {
            httplib::Client client(kv.second.host.c_str(), kv.second.http_port);
            client.set_connection_timeout(5, 0);
            client.set_read_timeout(120, 0);
            const json payload = { { "keep_manifest", keep_manifest } };
            const auto result = client.Post(
                    ("/models/" + model_id + "/reset").c_str(),
                    payload.dump(),
                    "application/json");
            if (result && result->status == 200) {
                try {
                    node_results[kv.first] = json::parse(result->body);
                } catch (...) {
                    node_results[kv.first] = { { "ok", true } };
                }
            } else {
                node_results[kv.first] = {
                    { "ok", false },
                    { "status", result ? result->status : 0 },
                };
            }
        }

        g_registry.clear_install_cluster_state(model_id, record);
        persist_registry();

        res.set_content(json({
            { "ok", true },
            { "model_id", model_id },
            { "keep_manifest", keep_manifest },
            { "nodes", node_results },
        }).dump(), "application/json");
    });

    // DELETE /models/{model_id} - Remove a registered model.
    svr.Delete(R"(/models/([^/]+))", [](const httplib::Request & req, httplib::Response & res) {
        const std::string model_id = req.matches[1];
        if (g_registry.remove(model_id)) {
            persist_registry();
            res.set_content(json({ { "ok", true } }).dump(), "application/json");
            return;
        }
        res.status = 404;
        res.set_content(json({ { "error", "model not registered" } }).dump(), "application/json");
    });

    if (!g_model_path.empty()) {
        fprintf(stderr, "orchestrator: local model=%s models_dir=%s\n",
                g_model_path.c_str(), g_models_dir.c_str());
    } else {
        fprintf(stderr, "orchestrator: layer-first mode models_dir=%s\n", g_models_dir.c_str());
    }
    fprintf(stderr, "orchestrator: listening on %s:%d (dynamic layer planner)\n", bind_host.c_str(), port);

    if (!svr.listen(bind_host.c_str(), port)) {
        fprintf(stderr, "orchestrator: failed to bind\n");
        return 1;
    }

    return 0;
}
