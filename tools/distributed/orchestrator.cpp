#include "dist_common.h"
#include "dist_rss_probe.h"
#include "layer_planner.h"
#include "memory_estimator.h"
#include "model_catalog.h"
#include "orchestrator/model_registry.h"
#include "orchestrator/registry_persistence.h"
#include "orchestrator/manifest_builder/manifest_builder.h"
#include "orchestrator/layout_planner/layout_planner.h"
#include "orchestrator/layout_policy/layout_policy.h"
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
#include "runtime/runtime_role_planner.h"
#include "runtime/runtime_graph.h"
#include "runtime/runtime_role.h"
#include "runtime/runtime_install_planning.h"
#include "dist_process.h"
#include "runtime_debug/perf_trace.h"
#include "runtime_debug/node_log_export.h"
#include "runtime_debug/perf_trace_export.h"

#include "httplib.h"
#include "nlohmann/json.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <map>
#include <mutex>
#include <optional>
#include <random>
#include <set>
#include <sstream>
#include <string>
#include <future>
#include <thread>
#include <vector>

#if !defined(_WIN32)
#include <unistd.h>
#endif

using json = nlohmann::json;

static constexpr const char * DEFAULT_GENERATE_PROMPT = "Tell me a joke";

struct dist_session {
    std::string session_id;
    std::string model;
    runtime_graph runtime;
    std::vector<dist_pipeline_stage> pipeline;
    std::string entry_host;
    int entry_ctrl_port = 0;
    int entry_layer_end = 0;
    bool active         = false;
    int generate_count  = 0;
    int configure_count = 0;
    int64_t created_at_ms = 0;
    // Task 19 Phase 3: draft placed on the `final` role (see
    // TASK_19_SPECULATIVE_PIPELINE_STUDY.md SA). The draft is always the
    // same fixed small model at a known URL -- no registry entry needed,
    // setup_runtime_graph just tells whichever node lands `final` to fetch
    // it directly (POST /draft/fetch) once that node is known, in parallel
    // with the primary model's per-stage downloads.
    // speculative_draft_model_path is the resolved local path on the final
    // node, filled in once that fetch completes (session-create blocks on
    // it, even if the primary model's layers were already cached and its
    // own loop was instant -- the draft is not optional once requested).
    std::string speculative_draft_model_url;
    std::string speculative_draft_model_path;
    int         speculative_draft_k = 4;
    // Sampling settings for this session, forwarded to whichever node ends up
    // sampling. Defaults reproduce the greedy chain the runtime used before
    // these were configurable.
    float        temp           = 0.0f;
    int          top_k          = 1;
    float        top_p          = 1.0f;
    float        min_p          = 0.0f;
    float        repeat_penalty = 1.0f;
    int          repeat_last_n  = 64;
    unsigned int seed           = 0xFFFFFFFF;
    // KV context this session's workers were built with. Kept so the
    // reported limit matches what the pipeline can actually hold.
    int          n_ctx          = 4096;
    // Client-supplied correlation id for create-time progress reporting.
    // Empty means the caller isn't watching, and no progress is recorded.
    std::string progress_id;

    json sampling_json() const {
        return {
            { "temp", temp },
            { "top_k", top_k },
            { "top_p", top_p },
            { "min_p", min_p },
            { "repeat_penalty", repeat_penalty },
            { "repeat_last_n", repeat_last_n },
            { "seed", seed },
        };
    }
};

static json session_debug_json(const dist_session & session) {
    json workers = json::array();
    for (const auto & stage : session.pipeline) {
        httplib::Client cli(stage.host.c_str(), stage.http_port);
        cli.set_connection_timeout(3, 0);
        cli.set_read_timeout(5, 0);
        json node_status = json::object();
        if (const auto res = cli.Get("/status")) {
            if (res->status == 200) {
                try {
                    node_status = json::parse(res->body);
                } catch (...) {}
            }
        }
        workers.push_back({
            { "node_id", stage.node_id },
            { "role", dist_role_name(stage.role) },
            { "host", stage.host },
            { "port", stage.http_port },
            { "status", node_status },
        });
    }
    return {
        { "session_id", session.session_id },
        { "model", session.model },
        { "active", session.active },
        { "generate_count", session.generate_count },
        { "configure_count", session.configure_count },
        { "workers", workers },
    };
}

static std::mutex g_mu;
static std::map<std::string, dist_node_info> g_nodes;
static std::map<std::string, dist_session> g_sessions;

// A node counts as online only while its heartbeats keep arriving. node_agent
// re-registers every 5s, so silence this long means the machine is actually
// gone rather than briefly busy.
//
// Until 2026-07-28 `online` was set true on registration and never set false
// anywhere in the tree, and `last_seen` was recorded but read by nothing. A
// stopped node therefore stayed "online" until the orchestrator itself
// restarted. That was not cosmetic: coverage polls kept going to a machine
// that was not there, came back with its layers marked missing, and every
// model placed on it read as DEGRADED -- which is how a perfectly healthy
// cluster kept looking broken, and why the same false alarm was investigated
// twice as if it were data loss. It also made layout_policy's `unavailable`
// verdict unreachable: the branch that exists specifically to protect an
// absent node's layers could never be taken, because no node was ever absent.
static constexpr int64_t NODE_OFFLINE_AFTER_S = 30;

static bool node_is_online(const dist_node_info & n) {
    return n.online && (dist_now_unix() - n.last_seen) <= NODE_OFFLINE_AFTER_S;
}

// Session-create progress, keyed by a client-supplied correlation id.
//
// /session/create is one blocking call that can take from seconds (warm) to
// minutes (cold, tens of GB of tensors to load), and the caller has no
// session_id to poll with until it returns -- hence a correlation id the
// client makes up and passes in. Omitting it disables tracking entirely, so
// existing callers (soak, ceiling measurement, benchmark runner) are
// unaffected.
struct session_create_progress {
    std::string phase;        // machine-readable stage key
    std::string detail;       // which node / what is happening
    int         step  = 0;    // 1-based within the phase, 0 = not countable
    int         total = 0;
    int64_t     started_ms = 0;
    int64_t     updated_ms = 0;
    bool        done   = false;
    bool        failed = false;
    std::string error;
    std::string session_id;   // filled once the session exists

    json to_json() const {
        return {
            { "phase", phase },
            { "detail", detail },
            { "step", step },
            { "total", total },
            { "started_ms", started_ms },
            { "updated_ms", updated_ms },
            { "elapsed_ms", updated_ms - started_ms },
            { "done", done },
            { "failed", failed },
            { "error", error },
            { "session_id", session_id },
        };
    }
};

static std::mutex g_progress_mu;
static std::map<std::string, session_create_progress> g_session_progress;

static int64_t progress_now_ms() {
    return std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::system_clock::now().time_since_epoch()).count();
}

// Drop finished entries the client stopped polling for; without this the map
// grows for the lifetime of the process.
static void progress_gc_locked() {
    const int64_t now = progress_now_ms();
    for (auto it = g_session_progress.begin(); it != g_session_progress.end();) {
        const bool stale = now - it->second.updated_ms > 10 * 60 * 1000;
        it = stale ? g_session_progress.erase(it) : std::next(it);
    }
}

static void progress_begin(const std::string & id) {
    if (id.empty()) {
        return;
    }
    std::lock_guard<std::mutex> lock(g_progress_mu);
    progress_gc_locked();
    session_create_progress p{};
    p.phase      = "queued";
    p.started_ms = progress_now_ms();
    p.updated_ms = p.started_ms;
    g_session_progress[id] = p;
}

static void progress_set(
        const std::string & id,
        const std::string & phase,
        const std::string & detail = "",
        int step = 0,
        int total = 0) {
    if (id.empty()) {
        return;
    }
    std::lock_guard<std::mutex> lock(g_progress_mu);
    const auto it = g_session_progress.find(id);
    if (it == g_session_progress.end()) {
        return;
    }
    it->second.phase      = phase;
    it->second.detail     = detail;
    it->second.step       = step;
    it->second.total      = total;
    it->second.updated_ms = progress_now_ms();
}

static void progress_finish(const std::string & id, const std::string & session_id) {
    if (id.empty()) {
        return;
    }
    std::lock_guard<std::mutex> lock(g_progress_mu);
    const auto it = g_session_progress.find(id);
    if (it == g_session_progress.end()) {
        return;
    }
    it->second.phase      = "ready";
    it->second.detail.clear();
    it->second.done       = true;
    it->second.session_id = session_id;
    it->second.updated_ms = progress_now_ms();
}

static void progress_fail(const std::string & id, const std::string & error) {
    if (id.empty()) {
        return;
    }
    std::lock_guard<std::mutex> lock(g_progress_mu);
    const auto it = g_session_progress.find(id);
    if (it == g_session_progress.end()) {
        return;
    }
    it->second.phase      = "failed";
    it->second.done       = true;
    it->second.failed     = true;
    it->second.error      = error;
    it->second.updated_ms = progress_now_ms();
}
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
static std::vector<dist_layer_assignment> assignments_from_desired_layout(
        const desired_model_layout & desired,
        const std::map<std::string, dist_node_info> & node_map,
        int n_layers);
static model_memory_requirements get_model_memory_for_record(
        const dist_model_record & record,
        int32_t n_ctx);
static std::optional<runtime_install_node_map> runtime_install_nodes_for_layout(
        const dist_model_record & record,
        const desired_model_layout & desired,
        const std::map<std::string, dist_node_info> & node_map);
static bool store_runtime_plan_for_layout(
        const std::string & model_id,
        const desired_model_layout & desired,
        const std::map<std::string, dist_node_info> & node_map);

static bool refresh_model_coverage(
        const std::string & model_id,
        const std::set<std::string> & online_nodes,
        dist_model_record * out = nullptr) {
    std::map<std::string, dist_node_info> node_map;
    {
        std::lock_guard<std::mutex> lock(g_mu);
        for (const std::string & nid : online_nodes) {
            const auto it = g_nodes.find(nid);
            if (it != g_nodes.end()) {
                node_map[nid] = it->second;
            }
        }
    }
    const dist_model_record * rec = g_registry.find(model_id);
    std::optional<runtime_install_node_map> runtime_nodes;
    if (rec != nullptr && rec->layout.has_value()) {
        runtime_nodes = runtime_install_nodes_for_layout(*rec, rec->layout->desired, node_map);
    }
    const runtime_install_node_map * runtime_ptr =
            runtime_nodes.has_value() ? &*runtime_nodes : nullptr;
    return g_registry.refresh_coverage(model_id, online_nodes, out, runtime_ptr);
}

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
    const bool perf_on = perf_trace_enabled();
    const std::string trace_id = perf_make_install_trace_id(job_id, model_id);
    if (perf_on) {
        perf_trace_set_component("orchestrator");
        perf_trace_set_node_id("orchestrator");
        perf_trace_begin_install(trace_id, "install");
    }

    if (nodes.empty()) {
        if (perf_on) {
            perf_emit_install_instant("INSTALL_FAILED", "plan", nullptr, nullptr, 0, "{\"error\":\"no online nodes\"}");
            perf_trace_end_install();
        }
        set_sync_job_state(job_id, "failed", "no online nodes");
        return;
    }

    if (plan.operations.empty()) {
        if (perf_on) {
            perf_emit_install_instant(
                    "INSTALL_FULL_REUSE",
                    "reuse",
                    "*",
                    "cluster",
                    0,
                    "{\"operation_count\":0}");
            perf_trace_end_install();
        }
        set_sync_job_state(job_id, "completed");
        return;
    }

    {
        char attrs[256];
        std::snprintf(
                attrs,
                sizeof(attrs),
                "{\"operation_count\":%d,\"total_download_bytes\":%llu}",
                plan.operation_count,
                static_cast<unsigned long long>(plan.total_download_bytes));
        if (perf_on) {
            perf_emit_install_instant("INSTALL_PLAN", "plan", nullptr, nullptr, plan.total_download_bytes, attrs);
        }
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
            { "trace_id", trace_id },
            { "perf_trace", perf_on },
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
        if (perf_on) {
            perf_emit_install_instant("INSTALL_FAILED", "plan", nullptr, nullptr, 0, "{\"error\":\"dispatch failed\"}");
            perf_trace_end_install();
        }
        set_sync_job_state(job_id, "failed", "failed to dispatch to any node");
        return;
    }

    set_sync_job_state(job_id, "running");

    perf_install_span install_wait_span("INSTALL_EXECUTE_WAIT", "plan");

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
                {
                    perf_install_span reconcile_span("INSTALL_RECONCILE", "plan");
                    poll_installed_layers_from_nodes(model_id, actual, online_nodes);
                dist_model_record * record = g_registry.find(model_id);
                if (record) {
                    g_registry.apply_actual(model_id, actual, online_nodes, record);
                    refresh_model_coverage(model_id, online_nodes, record);
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
                }
                if (perf_on) {
                    perf_emit_install_instant("INSTALL_COMPLETED", "plan", nullptr, nullptr, 0, nullptr);
                    perf_trace_end_install();
                }
            } else {
                if (perf_on) {
                    perf_emit_install_instant("INSTALL_FAILED", "plan", nullptr, nullptr, 0, "{\"error\":\"node job failed\"}");
                    perf_trace_end_install();
                }
                set_sync_job_state(job_id, "failed", "one or more node jobs failed");
            }
            return;
        }
    }

    if (perf_on) {
        perf_emit_install_instant("INSTALL_FAILED", "plan", nullptr, nullptr, 0, "{\"error\":\"timeout\"}");
        perf_trace_end_install();
    }
    set_sync_job_state(job_id, "failed", "sync job timed out");
}

static std::map<std::string, dist_node_info> collect_online_nodes_copy() {
    std::map<std::string, dist_node_info> nodes_copy;
    std::lock_guard<std::mutex> lock(g_mu);
    for (const auto & kv : g_nodes) {
        if (node_is_online(kv.second)) {
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
    g_registry.apply_actual(model_id, actual, online_nodes, record);
    refresh_model_coverage(model_id, online_nodes, record);

    record = g_registry.find(model_id);
    if (!record || !record->coverage.has_value()) {
        return false;
    }

    // Plan against the registry's merged view, not the raw poll. The poll only
    // contains nodes that answered, so using it directly reintroduces exactly
    // what apply_actual() was changed to prevent: an offline node's layers look
    // absent and get scheduled for re-download onto the machine that already
    // has them (gemma-3-1b's 320 MB embedding, measured 2026-07-28).
    const actual_model_layout & actual_for_plan =
            record->actual.has_value() ? *record->actual : actual;

    const coverage_report & coverage_for_plan = layout_override
            ? compute_coverage(*layout_override, actual_for_plan, online_nodes)
            : *record->coverage;

    std::map<std::string, dist_node_info> node_map;
    for (const std::string & node_id : online_nodes) {
        std::lock_guard<std::mutex> lock(g_mu);
        const auto it = g_nodes.find(node_id);
        if (it != g_nodes.end() && node_is_online(it->second)) {
            node_map[node_id] = it->second;
        }
    }

    const std::optional<runtime_install_node_map> runtime_nodes =
            runtime_install_nodes_for_layout(*record, *target, node_map);
    const runtime_install_node_map * runtime_nodes_ptr =
            runtime_nodes.has_value() ? &*runtime_nodes : nullptr;

    const auto built = build_install_plan(
            *record->manifest,
            *target,
            actual_for_plan,
            coverage_for_plan,
            resolve_model_source_url(*record),
            runtime_nodes_ptr,
            online_nodes);
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

// A model whose layers sit on a node that is not answering is UNAVAILABLE, not
// broken. "Repairing" it means fetching that node's share from the internet
// again and spreading it over whoever is left: correct when the machine is
// gone for good, pure waste when it is merely switched off -- and the second
// case is far more common in a home cluster.
//
// So the calls that would move data refuse by default and name the machine to
// turn on instead. `confirm_unavailable: true` is the caller stating that node
// is not coming back; the dashboard sends it from its confirm dialog.
struct unavailable_advice {
    bool                     applies = false;
    std::vector<std::string> offline_nodes;
    int                      unavailable_layers = 0;
};

static unavailable_advice describe_unavailable(const dist_model_record & record) {
    unavailable_advice out;
    if (!record.coverage.has_value() ||
            record.coverage->state != coverage_state::unavailable) {
        return out;
    }
    out.applies = true;
    out.unavailable_layers = record.coverage->unavailable_layers;

    // compute_coverage already worked out which machines those layers are on.
    out.offline_nodes = record.coverage->unavailable_nodes;
    return out;
}

static json unavailable_advice_json(
        const unavailable_advice & advice,
        const std::string & model_id) {
    std::string nodes;
    for (size_t i = 0; i < advice.offline_nodes.size(); ++i) {
        nodes += (i ? ", " : "") + advice.offline_nodes[i];
    }
    if (nodes.empty()) {
        nodes = "(unknown)";
    }

    return json{
        { "state", "UNAVAILABLE" },
        { "offline_nodes", advice.offline_nodes },
        { "unavailable_layers", advice.unavailable_layers },
        { "message",
          "Model is UNAVAILABLE, not damaged: " + std::to_string(advice.unavailable_layers) +
          " layer(s) are on " + nodes + ", which is not answering. Those layers are "
          "still on that machine's disk. Start it and the model returns on its own, "
          "with nothing transferred." },
        { "recommended_action", "start " + nodes + ", then re-check /models/" + model_id + "/coverage" },
        { "if_node_is_gone_for_good",
          "Proceeding re-downloads those layers from the internet and spreads them "
          "over the nodes that are left. Only do this if " + nodes + " is not coming back." },
        { "command",
          "curl -X POST <orchestrator>/models/" + model_id +
          "/optimize -H 'Content-Type: application/json' -d '{\"confirm_unavailable\":true}'" },
    };
}

// Body flag by which a caller states the offline node is not coming back.
static bool body_confirms_unavailable(const std::string & body) {
    if (body.empty()) {
        return false;
    }
    try {
        const json j = json::parse(body);
        return j.value("confirm_unavailable", false);
    } catch (...) {
        return false;
    }
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
    g_registry.apply_actual(model_id, actual, online_nodes);
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
    g_registry.apply_actual(model_id, actual, online_nodes, record);

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
        refresh_model_coverage(model_id, online_nodes, record);
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
                        refresh_model_coverage(model_id, online_nodes, record);
                    }
                }
            }
        }
    } else if (record && record->layout.has_value()) {
        refresh_model_coverage(model_id, online_nodes, record);
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

// Previously a destroyed session's pipeline-stage workers were only ever
// killed lazily -- inside the NEXT session's start_worker() ->
// stop_worker_for_role() call for the same node+role -- meaning a destroyed
// session's worker sat around holding memory/CPU indefinitely until some
// later, unrelated session happened to reconfigure that exact role. This
// contaminated later measurements with real resource contention (see
// docs/bench/2026-07-23_g1_ceiling/G1_CEILING_REPORT.md for the concrete
// case: repeated ceiling measurements on the same model swung from 96% to
// 43% purely from a stale prior session's worker still resident on node-a).
// Best-effort and fire-and-forget: destroy already succeeded from the
// caller's point of view (session removed from g_sessions) by the time this
// runs; a node being briefly unreachable here just means its worker gets
// cleaned up on the next /configure instead, same as before this fix.
static void stop_session_workers_async(const dist_session & session) {
    std::thread([pipeline = session.pipeline]() {
        for (const auto & stage : pipeline) {
            const std::string role_str = dist_role_name(stage.role);
            if (role_str == "unconfigured") {
                continue;
            }
            httplib::Client cli(stage.host.c_str(), stage.http_port);
            cli.set_connection_timeout(3, 0);
            cli.set_read_timeout(5, 0);
            cli.Post("/worker/stop", json({ { "role", role_str } }).dump(), "application/json");
        }
    }).detach();
}

// How many times in a row a model may be relaid out without reaching READY
// before this stops trying. Guards the feedback loop described below: a
// relayout that does not fix the model leaves it non-READY, which qualifies it
// for another relayout on the next cluster change, and so on.
static constexpr int OPTIMIZE_FUTILE_ATTEMPT_LIMIT = 2;

static std::mutex g_optimize_attempt_mu;
static std::map<std::string, int> g_optimize_attempts;

static void trigger_cluster_optimization_async() {
    std::thread([]() {
        // Decide from freshly polled coverage, never from whatever was
        // persisted before the last shutdown.
        //
        // This runs on node registration, and the first node to come back
        // after a restart triggers it while the others are still starting.
        // Their layers then look absent, every model looks broken, and the
        // optimizer "fixes" it by moving layers onto whichever node happens
        // to be up -- deleting the copies elsewhere. That is how a healthy
        // cluster lost data across a restart (gemma-3-1b, 2026-07-24), and
        // how layouts collapsed onto a single node earlier the same day.
        //
        // Re-polling per model costs one round trip per node and makes the
        // decision reflect the cluster as it is now.
        for (const auto & model : g_registry.list()) {
            if (model.manifest.has_value()) {
                sync_model_state_from_cluster(model.model_id, false);
            }
        }

        // Read the online set once. The policy's first question is whether the
        // layout's own nodes are up, and re-reading per model would let the
        // answer change halfway through the sweep -- so some models would be
        // judged against a three-node cluster and others against two.
        std::set<std::string> online_nodes;
        {
            std::lock_guard<std::mutex> lock(g_mu);
            for (const auto & entry : g_nodes) {
                if (node_is_online(entry.second)) {
                    online_nodes.insert(entry.first);
                }
            }
        }

        const auto models = g_registry.list();
        for (const auto & model : models) {
            if (!model.manifest.has_value()) {
                continue;
            }

            // What may be done to this layout is decided by one pure function
            // (Task 24) rather than by a stack of guards accumulated one
            // incident at a time. Each guard here was correct about the case
            // that produced it and silent about the ones it didn't, which is
            // why data loss kept recurring in a slightly different shape.
            layout_policy_input in;
            in.has_stored_layout = model.layout.has_value() &&
                    !model.layout->desired.placements.empty();
            if (in.has_stored_layout) {
                for (const auto & p : model.layout->desired.placements) {
                    in.layout_nodes.insert(p.node_id);
                }
            }
            in.online_nodes = online_nodes;
            // Every model with a manifest was re-polled at the top of this
            // sweep, which is the whole reason that loop exists.
            in.coverage_fresh = true;
            if (model.coverage.has_value()) {
                in.coverage         = model.coverage->state;
                in.missing_layers   = model.coverage->missing_layers;
                in.corrupted_layers = model.coverage->corrupted_layers;
                in.total_layers     = model.coverage->total_layers;
                in.ready_layers     = model.coverage->ready_layers;
            }

            std::string reason;
            const layout_action action = decide_layout_action(in, reason);

            if (action == layout_action::keep) {
                // Nothing to do -- and if it is healthy, forget earlier futile
                // attempts so a later, genuine problem still gets its retries.
                if (in.coverage == coverage_state::ready) {
                    std::lock_guard<std::mutex> lock(g_optimize_attempt_mu);
                    g_optimize_attempts.erase(model.model_id);
                }
                continue;
            }

            if (action == layout_action::unavailable) {
                fprintf(stderr, "orchestrator: %s layout left alone -- %s\n",
                        model.model_id.c_str(), reason.c_str());
                continue;
            }

            if (action == layout_action::repair) {
                // A data problem, not a placement problem: the fix is to move
                // blobs toward the stored layout, never to choose a new one.
                // Not started here, because repair plans contain DELETEs and a
                // background sweep is the wrong place to decide to delete
                // anything. Surfaced for the Models screen's repair action.
                fprintf(stderr,
                        "orchestrator: %s needs repair against its stored layout (%s) -- "
                        "not relaying out; run repair to fix the data\n",
                        model.model_id.c_str(), reason.c_str());
                continue;
            }

            // Only `recompute` reaches a relayout, and the policy makes that
            // rare by design: no stored layout, or a stored one that cannot
            // produce a runnable pipeline.
            {
                std::lock_guard<std::mutex> lock(g_optimize_attempt_mu);
                const int attempts = g_optimize_attempts[model.model_id];
                if (attempts >= OPTIMIZE_FUTILE_ATTEMPT_LIMIT) {
                    fprintf(stderr,
                            "orchestrator: %s still not READY after %d relayout attempts -- "
                            "not trying again automatically\n",
                            model.model_id.c_str(), attempts);
                    continue;
                }
                g_optimize_attempts[model.model_id] = attempts + 1;
            }

            fprintf(stderr, "orchestrator: %s relayout -- %s\n",
                    model.model_id.c_str(), reason.c_str());

            const optimization_result result = optimize_registered_model(model.model_id);
            if (result.decision == optimizer_decision::rebalance) {
                coordinate_rebalance_pipeline(model.model_id);
            }
        }
    }).detach();
}

static std::string configure_worker_artifact(const std::string & artifact) {
    return artifact.empty() ? "layer-store" : artifact;
}

static void perf_attach_trace(json & body) {
    if (!perf_trace_enabled()) {
        return;
    }
    std::string trace_id;
    std::string phase;
    int32_t token_idx = -1;
    if (perf_trace_get_context(trace_id, phase, token_idx) && !trace_id.empty()) {
        body["trace_id"]   = trace_id;
        body["perf_trace"] = true;
    }
}

static bool prepare_runtime_node(
        const dist_node_info & node,
        const json & body,
        std::string & worker_gguf_out,
        std::string & err,
        int timeout_ms = 600000) {
    const std::string role = body.value("role", body.value("runtime_role", "pipeline"));
    perf_session_span prep_span("SESSION_PREPARE_RUNTIME", node.node_id.c_str(), role.c_str());

    httplib::Client cli(node.host.c_str(), node.http_port);
    cli.set_connection_timeout(5, 0);
    cli.set_read_timeout(timeout_ms / 1000, (timeout_ms % 1000) * 1000);

    json req_body = body;
    perf_attach_trace(req_body);
    const auto res = cli.Post("/runtime/prepare", req_body.dump(), "application/json");
    if (!res) {
        err = "no response from " + node.node_id + " prepare at " + node.host + ":" +
              std::to_string(node.http_port);
        return false;
    }
    if (res->status != 200) {
        try {
            const json j = json::parse(res->body);
            err = node.node_id + " prepare: " + j.value("error", res->body);
        } catch (...) {
            err = node.node_id + " prepare HTTP " + std::to_string(res->status);
        }
        return false;
    }

    try {
        const json j = json::parse(res->body);
        if (!j.value("runtime_ready", false)) {
            err = node.node_id + ": " + j.value("error", "runtime prepare failed");
            return false;
        }
        worker_gguf_out = j.value("worker_gguf", "");
        const std::string rt_role = body.value("runtime_role", "");
        const bool bind_ready = j.value("tensors_ready", false) &&
                worker_gguf_out.empty() &&
                j.value("bind_source", "") == "layer_store";
        if (worker_gguf_out.empty() && rt_role != "sampler" && !bind_ready) {
            err = node.node_id + ": prepare returned empty worker_gguf";
            return false;
        }
        if (rt_role == "tokenizer" && !j.value("tokenizer_ready", false)) {
            err = node.node_id + ": prepare tokenizer not ready: " +
                  j.value("error", "tokenizer shell failed");
            return false;
        }
        if (rt_role == "embedding" && !j.value("embedding_ready", false)) {
            err = node.node_id + ": prepare embedding not ready: " +
                  j.value("error", "embedding shell failed");
            return false;
        }
        if (rt_role == "output_head" && !j.value("output_ready", false)) {
            err = node.node_id + ": prepare output not ready: " +
                  j.value("error", "output shell failed");
            return false;
        }
        return true;
    } catch (...) {
        err = node.node_id + ": invalid prepare response";
        return false;
    }
}

static bool configure_node(
        const dist_node_info & node,
        const json & body,
        std::string & err,
        int timeout_ms = 30000) {
    const std::string role = body.value("role", "pipeline");
    perf_session_span cfg_span("SESSION_CONFIGURE_NODE", node.node_id.c_str(), role.c_str());

    httplib::Client cli(node.host.c_str(), node.http_port);
    cli.set_connection_timeout(5, 0);
    cli.set_read_timeout(timeout_ms / 1000, (timeout_ms % 1000) * 1000);

    json req_body = body;
    perf_attach_trace(req_body);
    const auto res = cli.Post("/configure", req_body.dump(), "application/json");
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

static bool wait_node_worker_ready(
        const dist_node_info & node,
        const dist_node_role role,
        std::string & err,
        int timeout_ms = 300000) {
    const std::string role_name = dist_role_name(role);
    perf_session_span ready_span("SESSION_READY_WAIT", node.node_id.c_str(), role_name.c_str());

    httplib::Client cli(node.host.c_str(), node.http_port);
    cli.set_connection_timeout(5, 0);
    cli.set_read_timeout(5, 0);

    const auto start = std::chrono::steady_clock::now();
    while (true) {
        const auto res = cli.Get("/status");
        if (res && res->status == 200) {
            try {
                const json j = json::parse(res->body);
                const std::string role_name = dist_role_name(role);
                const std::string state =
                        j.value("worker_states", json::object()).value(role_name, "STARTING");
                if (state == "READY") {
                    return true;
                }
                if (state == "FAILED") {
                    err = node.node_id + " " + role_name + " worker failed during readiness";
                    return false;
                }
            } catch (...) {
                // Status can race with process restarts; keep polling until timeout.
            }
        }

        const auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::steady_clock::now() - start).count();
        if (elapsed >= timeout_ms) {
            err = node.node_id + " " + std::string(dist_role_name(role)) + " worker READY timeout";
            return false;
        }
#if defined(_WIN32)
        Sleep(100);
#else
        usleep(100000);
#endif
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
        if (node_it == node_map.end() || !node_is_online(node_it->second)) {
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

    // Manifest-based estimate only; orchestrator never loads GGUF for memory sizing.
    {
        std::lock_guard<std::mutex> lock(g_mu);
        const auto * model = g_catalog.find_model(record.model_id);
        if (model) {
            return estimate_model_memory_from_catalog(*model, n_ctx);
        }
    }

    return {};
}

static std::optional<runtime_descriptor> runtime_descriptor_for_record(
        const dist_model_record & record,
        std::string & err) {
    if (!record.manifest.has_value() || record.manifest->n_layer == 0) {
        err = "manifest not ready";
        return std::nullopt;
    }

    const std::string arch_id = record.manifest->architecture.empty()
            ? "unknown.arch"
            : record.manifest->architecture + ".arch";
    runtime_descriptor desc = make_basic_text_generation_runtime_descriptor(
            record.model_id,
            arch_id,
            static_cast<int32_t>(record.manifest->n_layer));

    const runtime_descriptor_validation validation =
            validate_runtime_descriptor(desc);
    if (!validation.ok()) {
        err = validation.errors.empty() ?
                "runtime descriptor validation failed" :
                validation.errors.front();
        return std::nullopt;
    }
    return desc;
}

static std::optional<runtime_install_node_map> runtime_install_nodes_for_layout(
        const dist_model_record & record,
        const desired_model_layout & desired,
        const std::map<std::string, dist_node_info> & node_map) {
    if (record.stored_runtime_graph.has_value()) {
        const runtime_install_node_map map =
                runtime_install_node_map_from_graph(*record.stored_runtime_graph);
        if (map.valid()) {
            return map;
        }
    }
    if (record.stored_runtime_install_nodes.has_value() &&
            record.stored_runtime_install_nodes->valid()) {
        return *record.stored_runtime_install_nodes;
    }
    if (!record.manifest.has_value() || record.manifest->n_layer == 0) {
        return std::nullopt;
    }
    const int n_layers = static_cast<int>(record.manifest->n_layer);
    const std::vector<dist_layer_assignment> assignments =
            assignments_from_desired_layout(desired, node_map, n_layers);
    if (assignments.empty()) {
        return std::nullopt;
    }
    const model_memory_requirements mem = get_model_memory_for_record(record, 512);
    if (!mem.valid()) {
        return std::nullopt;
    }
    std::string desc_err;
    const std::optional<runtime_descriptor> desc =
            runtime_descriptor_for_record(record, desc_err);
    if (!desc.has_value()) {
        return std::nullopt;
    }
    const runtime_role_planner_result plan = dist_plan_runtime_graph(
            *desc, mem, assignments, node_map);
    if (!plan.success) {
        return std::nullopt;
    }
    const runtime_execution_graph_validation graph_validation =
            validate_runtime_graph_against_descriptor(*desc, plan.graph);
    if (!graph_validation.ok()) {
        return std::nullopt;
    }
    runtime_install_node_map map = runtime_install_node_map_from_graph(plan.graph);
    if (!map.valid()) {
        return std::nullopt;
    }
    return map;
}

static bool store_runtime_plan_for_layout(
        const std::string & model_id,
        const desired_model_layout & desired,
        const std::map<std::string, dist_node_info> & node_map) {
    const dist_model_record * record = g_registry.find(model_id);
    if (!record || !record->manifest.has_value() || record->manifest->n_layer == 0) {
        return false;
    }
    const int n_layers = static_cast<int>(record->manifest->n_layer);
    const std::vector<dist_layer_assignment> assignments =
            assignments_from_desired_layout(desired, node_map, n_layers);
    if (assignments.empty()) {
        return false;
    }
    const model_memory_requirements mem = get_model_memory_for_record(*record, g_n_ctx);
    if (!mem.valid()) {
        return false;
    }
    std::string desc_err;
    const std::optional<runtime_descriptor> desc =
            runtime_descriptor_for_record(*record, desc_err);
    if (!desc.has_value()) {
        fprintf(stderr,
                "orchestrator: runtime descriptor invalid for %s: %s\n",
                model_id.c_str(),
                desc_err.c_str());
        return false;
    }
    const runtime_role_planner_result plan = dist_plan_runtime_graph(
            *desc, mem, assignments, node_map);
    if (!plan.success) {
        return false;
    }
    const runtime_execution_graph_validation graph_validation =
            validate_runtime_graph_against_descriptor(*desc, plan.graph);
    if (!graph_validation.ok()) {
        fprintf(stderr,
                "orchestrator: runtime graph invalid for %s: %s\n",
                model_id.c_str(),
                graph_validation.errors.empty() ? "unknown" : graph_validation.errors.front().c_str());
        return false;
    }
    runtime_install_node_map install_map = runtime_install_node_map_from_graph(plan.graph);
    if (!install_map.valid()) {
        return false;
    }
    fprintf(stderr,
            "orchestrator: runtime plan model=%s embedding=%s output=%s tokenizer=%s\n",
            model_id.c_str(),
            install_map.embedding_node.c_str(),
            install_map.output_head_node.c_str(),
            install_map.tokenizer_node.c_str());
    return g_registry.apply_runtime_plan(model_id, plan.graph, std::move(install_map), nullptr);
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
        if (!node_is_online(n)) {
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

namespace {

struct node_layer_poll_result {
    std::string node_id;
    bool ok = false;
    actual_model_layout layout;
};

// A node under this poll is frequently ALSO busy inside a concurrent
// /configure call for a different session, loading tens of GB of tensors --
// that can make it briefly too slow to answer this admin-plane GET within
// the per-attempt timeout even though its actual install state never
// changed. Observed 2026-07-23 (docs/bench/2026-07-23_g3_soak): a soak of
// back-to-back session churn hit this on 31/107 cycles, always as a clean
// "runtime coverage not ready" false negative, never a real state problem.
// One retry absorbed the transient in every case checked.
node_layer_poll_result poll_one_node_installed_layers(
        const dist_node_info & node,
        const std::string & model_id) {
    node_layer_poll_result r;
    r.node_id = node.node_id;

    const std::string path = "/installed-layers?model=" + model_id;
    for (int attempt = 0; attempt < 2; ++attempt) {
        httplib::Client client(node.host.c_str(), node.http_port);
        client.set_connection_timeout(3, 0);
        client.set_read_timeout(10, 0);

        const auto result = client.Get(path.c_str());
        if (result && result->status == 200) {
            try {
                const json body = json::parse(result->body);
                r.layout = actual_layout_from_node_response(node.node_id, body);
                r.ok = true;
                return r;
            } catch (...) {
                // fall through to retry
            }
        }
    }
    return r;
}

} // namespace

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

    // Poll every node concurrently -- sequential polling made one slow node
    // serialize (and thus multiply the timeout risk for) every other node's
    // check behind it, on every single session-create.
    std::vector<std::future<node_layer_poll_result>> futures;
    futures.reserve(nodes_copy.size());
    for (const auto & kv : nodes_copy) {
        if (!node_is_online(kv.second)) {
            continue;
        }
        const dist_node_info node = kv.second;
        futures.push_back(std::async(std::launch::async,
                [node, model_id]() { return poll_one_node_installed_layers(node, model_id); }));
    }

    for (auto & fut : futures) {
        const node_layer_poll_result r = fut.get();
        if (r.ok) {
            reports.push_back(r.layout);
            online_nodes.insert(r.node_id);
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
        std::string & err);

static bool setup_runtime_graph(
        const std::string & session_id,
        int n_layers,
        const std::vector<dist_layer_assignment> & assignments,
        const std::map<std::string, dist_node_info> & node_map,
        const model_memory_requirements & mem,
        dist_session & session,
        std::string & err);

static bool external_embedding_enabled() {
    const char * v = std::getenv("DIST_EXTERNAL_EMBEDDING");
    if (v == nullptr) {
        return true;
    }
    return v[0] == '1' && v[1] == '\0';
}

static bool external_output_enabled() {
    const char * v = std::getenv("DIST_EXTERNAL_OUTPUT");
    if (v == nullptr) {
        return true;
    }
    return v[0] == '1' && v[1] == '\0';
}

static bool prepare_service_role(
        runtime_role_assignment & assign,
        const json & prep,
        const std::map<std::string, dist_node_info> & node_map,
        std::string & artifact,
        std::string & err,
        const int timeout_ms = 600000) {
    const dist_node_info * node = find_node(node_map, assign.node_id);
    if (!node) {
        err = "service node missing: " + assign.node_id;
        return false;
    }
    if (prepare_runtime_node(*node, prep, artifact, err, timeout_ms)) {
        return true;
    }
    return false;
}

static bool setup_runtime_graph(
        const std::string & session_id,
        int n_layers,
        const std::vector<dist_layer_assignment> & assignments,
        const std::map<std::string, dist_node_info> & node_map,
        const model_memory_requirements & mem,
        dist_session & session,
        std::string & err) {
    (void) n_layers;

    runtime_role_planner_result plan{};
    const dist_model_record * record = g_registry.find(session.model);
    if (record == nullptr) {
        err = "model record missing";
        return false;
    }
    std::string desc_err;
    const std::optional<runtime_descriptor> desc =
            runtime_descriptor_for_record(*record, desc_err);
    if (!desc.has_value()) {
        err = "runtime descriptor: " + desc_err;
        return false;
    }
    if (record != nullptr && record->stored_runtime_graph.has_value()) {
        plan.success = true;
        plan.graph   = *record->stored_runtime_graph;
    } else {
        plan = dist_plan_runtime_graph(
                *desc, mem, assignments, node_map);
    }
    if (!plan.success) {
        err = "runtime graph: " + plan.error;
        return false;
    }
    const runtime_execution_graph_validation graph_validation =
            validate_runtime_graph_against_descriptor(*desc, plan.graph);
    if (!graph_validation.ok()) {
        err = "runtime graph validation: " +
                (graph_validation.errors.empty() ? "unknown" : graph_validation.errors.front());
        return false;
    }
    session.runtime = std::move(plan.graph);

    const auto stage_ptrs = session.runtime.pipeline_stages();
    if (stage_ptrs.empty()) {
        err = "runtime graph has no pipeline stages";
        return false;
    }
    const std::string first_pipeline_node = stage_ptrs.front()->node_id;
    const std::string last_pipeline_node  = stage_ptrs.back()->node_id;

    std::set<std::string> touched_nodes;
    for (const auto & a : session.runtime.assignments) {
        touched_nodes.insert(a.node_id);
    }
    {
        perf_session_span shutdown_span("SESSION_SHUTDOWN_NODES", "cluster", "orchestrator");
        for (const auto & nid : touched_nodes) {
            const dist_node_info * node = find_node(node_map, nid);
            if (node) {
                shutdown_node(*node);
            }
        }
#if !defined(_WIN32)
        usleep(300000);
#endif
    }

    if (const dist_model_record * record = g_registry.find(session.model)) {
        const std::string source_url = resolve_model_source_url(*record);

        for (auto & a : session.runtime.assignments) {
            if (!runtime_role_is_service(a.role)) {
                continue;
            }
            if (a.role == runtime_role::embedding && a.node_id == first_pipeline_node) {
                // The entry worker already owns token embedding tensors. Avoid
                // loading a duplicate embedding service on the same constrained
                // Docker node.
                continue;
            }
            if (a.role == runtime_role::output_head && a.node_id == last_pipeline_node) {
                // The final worker can execute output head locally when the
                // service is placed on the same boundary node.
                continue;
            }

            progress_set(session.progress_id, "services",
                    runtime_role_name(a.role) + " @ " + a.node_id);

            json prep = {
                { "session_id", session_id },
                { "model_id", session.model },
                { "runtime_role", runtime_role_name(a.role) },
                { "source_url", source_url },
            };
            std::string artifact;
            if (!prepare_service_role(a, prep, node_map, artifact, err, 600000)) {
                return false;
            }

            httplib::Client cli(a.host.c_str(), a.http_port);
            cli.set_connection_timeout(5, 0);
            cli.set_read_timeout(120, 0);

            if (a.role == runtime_role::tokenizer) {
                json cfg = {
                    { "session_id", session_id },
                    { "model_id", session.model },
                    { "worker_gguf", configure_worker_artifact(artifact) },
                };
                perf_attach_trace(cfg);
                const std::string svc_role = runtime_role_name(a.role);
                perf_session_span svc_span(
                        "SESSION_SERVICE_CONFIGURE",
                        a.node_id.c_str(),
                        svc_role.c_str());
                const auto res = cli.Post("/runtime/tokenizer/configure", cfg.dump(), "application/json");
                if (!res || res->status != 200) {
                    err = a.node_id + " tokenizer configure failed";
                    return false;
                }
                try {
                    const json j = json::parse(res->body);
                    if (!j.value("ok", false) || !j.value("tokenizer_ready", false)) {
                        err = a.node_id + " tokenizer configure: " +
                              j.value("error", "tokenizer not ready");
                        return false;
                    }
                } catch (...) {
                    err = a.node_id + " tokenizer configure: invalid response";
                    return false;
                }
            } else if (a.role == runtime_role::embedding) {
                json cfg = {
                    { "session_id", session_id },
                    { "model_id", session.model },
                    { "worker_gguf", configure_worker_artifact(artifact) },
                };
                perf_attach_trace(cfg);
                const std::string svc_role = runtime_role_name(a.role);
                perf_session_span svc_span(
                        "SESSION_SERVICE_CONFIGURE",
                        a.node_id.c_str(),
                        svc_role.c_str());
                const auto res = cli.Post("/runtime/embedding/configure", cfg.dump(), "application/json");
                if (!res || res->status != 200) {
                    err = a.node_id + " embedding configure failed";
                    return false;
                }
                try {
                    const json j = json::parse(res->body);
                    if (!j.value("ok", false) || !j.value("embedding_ready", false)) {
                        err = a.node_id + " embedding configure: " +
                              j.value("error", "embedding not ready");
                        return false;
                    }
                } catch (...) {
                    err = a.node_id + " embedding configure: invalid response";
                    return false;
                }
            } else if (a.role == runtime_role::output_head) {
                json cfg = {
                    { "session_id", session_id },
                    { "model_id", session.model },
                    { "worker_gguf", configure_worker_artifact(artifact) },
                    { "sampling", session.sampling_json() },
                };
                perf_attach_trace(cfg);
                const std::string svc_role = runtime_role_name(a.role);
                perf_session_span svc_span(
                        "SESSION_SERVICE_CONFIGURE",
                        a.node_id.c_str(),
                        svc_role.c_str());
                const auto res = cli.Post("/runtime/output/configure", cfg.dump(), "application/json");
                if (!res || res->status != 200) {
                    err = a.node_id + " output configure failed";
                    return false;
                }
                try {
                    const json j = json::parse(res->body);
                    if (!j.value("ok", false) || !j.value("output_ready", false)) {
                        err = a.node_id + " output configure: " +
                              j.value("error", "output not ready");
                        return false;
                    }
                } catch (...) {
                    err = a.node_id + " output configure: invalid response";
                    return false;
                }
            } else if (a.role == runtime_role::sampler) {
                json cfg = {
                    { "session_id", session_id },
                    { "model_id", session.model },
                    { "sampling", session.sampling_json() },
                };
                perf_attach_trace(cfg);
                const std::string svc_role = runtime_role_name(a.role);
                perf_session_span svc_span(
                        "SESSION_SERVICE_CONFIGURE",
                        a.node_id.c_str(),
                        svc_role.c_str());
                const auto res = cli.Post("/runtime/sampler/configure", cfg.dump(), "application/json");
                if (!res || res->status != 200) {
                    err = a.node_id + " sampler configure failed";
                    return false;
                }
            }
        }
    }

    const int pipe_base = 9100 + (int) (getpid() % 500) + 10;
    // Task 19 Phase 3: direct entry<->final link for draft-token delivery
    // (see TASK_19_SPECULATIVE_PIPELINE_STUDY.md SC). Reuses the same port
    // space as ctrl/peer, one slot past the last stage's peer_port.
    const bool draft_requested = !session.speculative_draft_model_url.empty();
    bool speculative = false;
    const int fa_port = pipe_base + (int) stage_ptrs.size() + 2;
    // Task 21.2: direct final->client token-return link (tc-link), one slot
    // past fa_port. Entry's node_agent listens (observability only for now
    // -- see TASK_21_PROVEN_PRACTICES_PLAN.md Item 2), final connects.
    const int tc_port = pipe_base + (int) stage_ptrs.size() + 3;
    const runtime_role_assignment * emb_assign = session.runtime.find_role(runtime_role::embedding);
    const runtime_role_assignment * out_assign = session.runtime.find_role(runtime_role::output_head);
    const bool external_embedding =
            external_embedding_enabled() && emb_assign != nullptr && emb_assign->node_id != first_pipeline_node;
    const bool external_output =
            external_output_enabled() && out_assign != nullptr && out_assign->node_id != last_pipeline_node;

    std::vector<dist_pipeline_stage> stages;
    stages.reserve(stage_ptrs.size());
    for (size_t i = 0; i < stage_ptrs.size(); ++i) {
        const runtime_role_assignment & ra = *stage_ptrs[i];
        const dist_node_info * node = find_node(node_map, ra.node_id);
        if (!node) {
            err = "pipeline node missing: " + ra.node_id;
            return false;
        }

        dist_pipeline_stage stage{};
        stage.node_id     = ra.node_id;
        stage.host        = node->host;
        stage.http_port   = node->http_port;
        stage.layer_start = ra.layer_start;
        stage.layer_end   = ra.layer_end;
        stage.score       = ra.score;

        if (i == 0) {
            stage.role      = DIST_ROLE_ENTRY;
            stage.ctrl_port = pipe_base + 1;
        } else if (i + 1 == stage_ptrs.size()) {
            stage.role      = DIST_ROLE_FINAL;
            stage.peer_port = pipe_base + (int) i + 1;
        } else {
            stage.role      = DIST_ROLE_MIDDLE;
            stage.peer_port = pipe_base + (int) i + 1;
        }
        stages.push_back(stage);
    }

    // A one-stage pipeline is not a supported configuration: the role loop
    // above assigns ENTRY on i == 0 and only reaches the FINAL branch for a
    // later stage, so a single stage ends up with no final at all. Everything
    // downstream then works against a pipeline that has no output head, no
    // sampler placement and no fa/tc endpoints -- which crashed the whole
    // orchestrator process when a layout concentrated every layer onto one
    // node (2026-07-24, the Task 21.4 revert; see docs/KNOWN_ISSUES.md).
    //
    // Failing here turns that into a clean, explainable 500 instead of taking
    // the process down. Supporting single-node inference properly is a
    // separate piece of work -- this only refuses to pretend it works.
    if (stages.size() < 2) {
        err = "layout produced a " + std::to_string(stages.size()) +
              "-stage pipeline; at least 2 stages are required. The stored "
              "layout for this model likely concentrates every layer on one "
              "node -- recompute it with POST /models/<id>/layout {\"force\":true} "
              "and repair coverage afterwards.";
        return false;
    }

    // Task 19 Phase 3: draft placement follows `final` (SA). The draft is
    // a fixed model at a known URL -- no registry entry, just tell the
    // final node to fetch it (POST /draft/fetch). Kicked off now so it
    // downloads in parallel with the primary model's per-stage prepare
    // loop below, and joined after that loop regardless of how fast it
    // ran (a cache hit on the primary model must not silently skip
    // speculation for this session; the draft is not optional once
    // requested, see F.2).
    std::future<std::pair<bool, std::string>> draft_future;
    if (draft_requested) {
        const dist_node_info * draft_node = find_node(node_map, stages.back().node_id);
        if (draft_node == nullptr) {
            fprintf(stderr, "orchestrator: draft node vanished, speculation disabled\n");
        } else {
            std::string filename = session.speculative_draft_model_url;
            const size_t slash = filename.find_last_of('/');
            if (slash != std::string::npos) {
                filename = filename.substr(slash + 1);
            }
            const json draft_req = {
                { "source_url", session.speculative_draft_model_url },
                { "filename", filename },
            };
            const dist_node_info draft_node_copy = *draft_node;
            draft_future = std::async(std::launch::async, [draft_node_copy, draft_req]() {
                httplib::Client cli(draft_node_copy.host.c_str(), draft_node_copy.http_port);
                cli.set_connection_timeout(5, 0);
                cli.set_read_timeout(1800, 0);
                const auto res = cli.Post("/draft/fetch", draft_req.dump(), "application/json");
                if (!res) {
                    return std::make_pair(false, std::string("no response from ") + draft_node_copy.node_id);
                }
                try {
                    const json j = json::parse(res->body);
                    if (res->status == 200 && j.value("ok", false)) {
                        return std::make_pair(true, j.value("path", std::string()));
                    }
                    return std::make_pair(false, j.value("error", res->body));
                } catch (...) {
                    return std::make_pair(false, std::string("bad response body"));
                }
            });
        }
    }

    const size_t n_stages = stages.size();
    std::vector<std::string> worker_ggufs(n_stages);

    for (size_t i = 0; i < n_stages; ++i) {
        const auto & stage = stages[i];
        const dist_node_info * node = find_node(node_map, stage.node_id);
        if (!node) {
            err = "node vanished: " + stage.node_id;
            return false;
        }

        json prep = {
            { "session_id", session_id },
            { "model_id", session.model },
            { "layer_start", stage.layer_start },
            { "layer_end", stage.layer_end },
            { "peer_bind", "0.0.0.0" },
            { "runtime_role", "pipeline_stage" },
            { "n_ctx", session.n_ctx },
        };
        if (external_embedding && i == 0) {
            prep["external_embedding"] = true;
        }
        if (external_output && stage.role == DIST_ROLE_FINAL) {
            prep["external_output"] = true;
        }
        if (const dist_model_record * record = g_registry.find(session.model)) {
            prep["source_url"] = resolve_model_source_url(*record);
        }
        if (stage.role == DIST_ROLE_FINAL) {
            prep["role"] = "final";
        } else if (stage.role == DIST_ROLE_MIDDLE) {
            prep["role"] = "middle";
        } else {
            prep["role"] = "entry";
            if (external_embedding) {
                prep["external_embedding"] = true;
            }
        }

        dist_rss_log_stage("prepare_runtime", stage.node_id.c_str());
        // The long pole on a cold session: this is where each node loads its
        // slice of tensors, up to tens of GB.
        progress_set(session.progress_id, "prepare_nodes", stage.node_id,
                (int) i + 1, (int) stages.size());
        if (!prepare_runtime_node(*node, prep, worker_ggufs[i], err, 600000)) {
            return false;
        }
    }

    if (draft_future.valid()) {
        progress_set(session.progress_id, "draft_fetch", "");
        const auto [ok, result] = draft_future.get();
        if (ok) {
            session.speculative_draft_model_path = result;
            speculative = true;
        } else {
            fprintf(stderr, "orchestrator: draft model prepare failed: %s (speculation disabled for session %s)\n",
                    result.c_str(), session_id.c_str());
        }
    }

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
            { "skip_materialize", true },
            { "worker_gguf", configure_worker_artifact(worker_ggufs[(size_t) ri]) },
            { "runtime_role", "pipeline_stage" },
        };
        if (external_embedding && ri == 0) {
            cfg["external_embedding"] = true;
        }
        if (external_output && stage.role == DIST_ROLE_FINAL) {
            cfg["external_output"] = true;
        }
        if (const dist_model_record * record = g_registry.find(session.model)) {
            cfg["source_url"] = resolve_model_source_url(*record);
        }
        if (stage.role == DIST_ROLE_FINAL) {
            cfg["role"] = "final";
            cfg["peer_port"] = stage.peer_port;
            // Only the final stage samples, so it is the only one that needs
            // these (the output service gets its own copy below when the
            // output head lives elsewhere).
            cfg["sampling"] = session.sampling_json();
            if (speculative) {
                cfg["draft_model"] = session.speculative_draft_model_path;
                cfg["draft_k"]     = session.speculative_draft_k;
                cfg["fa_host"]     = stages[0].host;
                cfg["fa_port"]     = fa_port;
                cfg["tc_host"]     = stages[0].host;
                cfg["tc_port"]     = tc_port;
            }
            if (external_output && out_assign != nullptr) {
                const std::string out_host =
                        !out_assign->endpoint.host.empty() ? out_assign->endpoint.host : out_assign->host;
                const int out_port =
                        out_assign->endpoint.port > 0 ? out_assign->endpoint.port : out_assign->http_port;
                cfg["output_service"] = {
                    { "scheme", out_assign->endpoint.scheme.empty() ? "http" : out_assign->endpoint.scheme },
                    { "host", out_host },
                    { "port", out_port },
                    { "node_id", out_assign->node_id },
                    { "role", runtime_role_name(out_assign->role) },
                };
            }
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
            if (speculative) {
                cfg["fa_port"] = fa_port;
                cfg["tc_port"] = tc_port;
            }
            if (external_embedding && emb_assign != nullptr) {
                const std::string emb_host =
                        !emb_assign->endpoint.host.empty() ? emb_assign->endpoint.host : emb_assign->host;
                const int emb_port =
                        emb_assign->endpoint.port > 0 ? emb_assign->endpoint.port : emb_assign->http_port;
                cfg["external_embedding"] = true;
                cfg["embedding_service"] = {
                    { "scheme", emb_assign->endpoint.scheme.empty() ? "http" : emb_assign->endpoint.scheme },
                    { "host", emb_host },
                    { "port", emb_port },
                    { "node_id", emb_assign->node_id },
                    { "role", runtime_role_name(emb_assign->role) },
                };
            }
        }

        progress_set(session.progress_id, "spawn_workers", stage.node_id,
                (int) ri + 1, (int) stages.size());
        if (!configure_node(*node, cfg, err, 300000)) {
            return false;
        }
#if !defined(_WIN32)
        usleep(ri == 0 ? 500000 : 200000);
#endif
    }

    for (size_t si = 0; si < stages.size(); ++si) {
        const auto & stage = stages[si];
        const dist_node_info * node = find_node(node_map, stage.node_id);
        if (!node) {
            err = "node vanished: " + stage.node_id;
            return false;
        }
        progress_set(session.progress_id, "await_workers", stage.node_id,
                (int) si + 1, (int) stages.size());
        if (!wait_node_worker_ready(*node, stage.role, err, 300000)) {
            return false;
        }
    }

    for (size_t i = 0; i < stage_ptrs.size(); ++i) {
        for (auto & a : session.runtime.assignments) {
            if (a.role == runtime_role::pipeline_stage && a.stage_index == (int) i) {
                a.ctrl_port = stages[i].ctrl_port;
                a.peer_port = stages[i].peer_port;
                break;
            }
        }
    }

    session.pipeline        = stages;
    session.entry_host      = stages[0].host;
    session.entry_ctrl_port = stages[0].ctrl_port;
    session.entry_layer_end = stages[0].layer_end;
    session.active          = true;
    session.configure_count = (int) session.runtime.assignments.size();
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
    std::vector<std::string> worker_ggufs(n_stages);

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

    // Materialize worker GGUF on each node (data plane), then configure workers.
    for (size_t i = 0; i < n_stages; ++i) {
        const auto & stage = stages[i];
        const dist_node_info * node = find_node(node_map, stage.node_id);
        if (!node) {
            err = "node vanished: " + stage.node_id;
            return false;
        }

        json prep = {
            { "session_id", session_id },
            { "model_id", session.model },
            { "layer_start", stage.layer_start },
            { "layer_end", stage.layer_end },
            { "peer_bind", "0.0.0.0" },
        };

        if (const dist_model_record * record = g_registry.find(session.model)) {
            prep["source_url"] = resolve_model_source_url(*record);
        }

        if (stage.role == DIST_ROLE_FINAL) {
            prep["role"] = "final";
        } else if (stage.role == DIST_ROLE_MIDDLE) {
            prep["role"] = "middle";
        } else {
            prep["role"] = "entry";
        }

        dist_rss_log_stage("prepare_runtime", stage.node_id.c_str());
        if (!prepare_runtime_node(*node, prep, worker_ggufs[i], err, 600000)) {
            return false;
        }
    }

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
            { "skip_materialize", true },
            { "worker_gguf", configure_worker_artifact(worker_ggufs[(size_t) ri]) },
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

        if (!configure_node(*node, cfg, err, 300000)) {
            return false;
        }

#if !defined(_WIN32)
        usleep(ri == 0 ? 500000 : 200000);
#endif
    }

    for (const auto & stage : stages) {
        const dist_node_info * node = find_node(node_map, stage.node_id);
        if (!node) {
            err = "node vanished: " + stage.node_id;
            return false;
        }
        if (!wait_node_worker_ready(*node, stage.role, err, 300000)) {
            return false;
        }
    }

    session.pipeline         = stages;
    session.entry_host       = stages[0].host;
    session.entry_ctrl_port  = stages[0].ctrl_port;
    session.entry_layer_end  = stages[0].layer_end;
    session.active           = true;
    session.configure_count  = (int) n_stages;
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

static bool recover_session_pipeline(dist_session & session, std::string & err) {
    if (session.pipeline.empty()) {
        err = "empty pipeline";
        return false;
    }

    std::map<std::string, dist_node_info> node_map;
    {
        std::lock_guard<std::mutex> lock(g_mu);
        node_map = g_nodes;
    }

    const runtime_role_assignment * emb_assign = session.runtime.find_role(runtime_role::embedding);
    const runtime_role_assignment * out_assign = session.runtime.find_role(runtime_role::output_head);
    const std::string first_pipeline_node = session.pipeline.front().node_id;
    const std::string last_pipeline_node  = session.pipeline.back().node_id;
    const bool external_embedding =
            external_embedding_enabled() && emb_assign != nullptr && emb_assign->node_id != first_pipeline_node;
    const bool external_output =
            external_output_enabled() && out_assign != nullptr && out_assign->node_id != last_pipeline_node;

    for (const auto & stage : session.pipeline) {
        const dist_node_info * node = find_node(node_map, stage.node_id);
        if (node == nullptr) {
            err = "pipeline recovery missing node: " + stage.node_id;
            return false;
        }
        shutdown_node(*node);
    }
#if !defined(_WIN32)
    usleep(300000);
#endif

    // Re-bind each worker's layer-store materialization before reconfiguring:
    // skip_materialize=true below requires a worker_gguf handle (see
    // configure_session_pipeline for the same two-step pattern). Recovery
    // previously set skip_materialize without ever calling
    // prepare_runtime_node, so every reconfigure failed with "worker_gguf
    // required when skip_materialize=true" and recovery could never succeed.
    const size_t n_stages = session.pipeline.size();
    std::vector<std::string> worker_ggufs(n_stages);
    for (size_t i = 0; i < n_stages; ++i) {
        const auto & stage = session.pipeline[i];
        const dist_node_info * node = find_node(node_map, stage.node_id);
        if (node == nullptr) {
            err = "pipeline recovery node vanished: " + stage.node_id;
            return false;
        }

        json prep = {
            { "session_id", session.session_id },
            { "model_id", session.model },
            { "layer_start", stage.layer_start },
            { "layer_end", stage.layer_end },
            { "peer_bind", "0.0.0.0" },
        };
        if (const dist_model_record * record = g_registry.find(session.model)) {
            prep["source_url"] = resolve_model_source_url(*record);
        }
        if (stage.role == DIST_ROLE_FINAL) {
            prep["role"] = "final";
        } else if (stage.role == DIST_ROLE_MIDDLE) {
            prep["role"] = "middle";
        } else {
            prep["role"] = "entry";
        }

        if (!prepare_runtime_node(*node, prep, worker_ggufs[i], err, 600000)) {
            err = "pipeline recovery prepare " + stage.node_id + ": " + err;
            return false;
        }
    }

    for (int ri = (int) session.pipeline.size() - 1; ri >= 0; --ri) {
        const auto & stage = session.pipeline[(size_t) ri];
        const dist_node_info * node = find_node(node_map, stage.node_id);
        if (node == nullptr) {
            err = "pipeline recovery node vanished: " + stage.node_id;
            return false;
        }

        json cfg = {
            { "session_id", session.session_id },
            { "model_id", session.model },
            { "layer_start", stage.layer_start },
            { "layer_end", stage.layer_end },
            { "peer_bind", "0.0.0.0" },
            { "skip_materialize", true },
            { "worker_gguf", configure_worker_artifact(worker_ggufs[(size_t) ri]) },
            { "runtime_role", "pipeline_stage" },
        };
        if (const dist_model_record * record = g_registry.find(session.model)) {
            cfg["source_url"] = resolve_model_source_url(*record);
        }
        if (external_embedding && ri == 0) {
            cfg["external_embedding"] = true;
        }
        if (external_output && stage.role == DIST_ROLE_FINAL) {
            cfg["external_output"] = true;
        }

        if (stage.role == DIST_ROLE_FINAL) {
            cfg["role"] = "final";
            cfg["peer_port"] = stage.peer_port;
            if (external_output && out_assign != nullptr) {
                const std::string out_host =
                        !out_assign->endpoint.host.empty() ? out_assign->endpoint.host : out_assign->host;
                const int out_port =
                        out_assign->endpoint.port > 0 ? out_assign->endpoint.port : out_assign->http_port;
                cfg["output_service"] = {
                    { "scheme", out_assign->endpoint.scheme.empty() ? "http" : out_assign->endpoint.scheme },
                    { "host", out_host },
                    { "port", out_port },
                    { "node_id", out_assign->node_id },
                    { "role", runtime_role_name(out_assign->role) },
                };
            }
        } else if (stage.role == DIST_ROLE_MIDDLE) {
            const auto & next = session.pipeline[(size_t) ri + 1];
            cfg["role"] = "middle";
            cfg["peer_port"] = stage.peer_port;
            cfg["next_host"] = next.host;
            cfg["next_port"] = next.peer_port;
        } else {
            const auto & next = session.pipeline[(size_t) ri + 1];
            cfg["role"] = "entry";
            cfg["ctrl_port"] = stage.ctrl_port;
            cfg["next_host"] = next.host;
            cfg["next_port"] = next.peer_port;
            cfg["next_is_final"] = (next.role == DIST_ROLE_FINAL);
            if (external_embedding && emb_assign != nullptr) {
                const std::string emb_host =
                        !emb_assign->endpoint.host.empty() ? emb_assign->endpoint.host : emb_assign->host;
                const int emb_port =
                        emb_assign->endpoint.port > 0 ? emb_assign->endpoint.port : emb_assign->http_port;
                cfg["external_embedding"] = true;
                cfg["embedding_service"] = {
                    { "scheme", emb_assign->endpoint.scheme.empty() ? "http" : emb_assign->endpoint.scheme },
                    { "host", emb_host },
                    { "port", emb_port },
                    { "node_id", emb_assign->node_id },
                    { "role", runtime_role_name(emb_assign->role) },
                };
            }
        }

        if (!configure_node(*node, cfg, err, 300000)) {
            err = "pipeline recovery configure " + stage.node_id + ": " + err;
            return false;
        }
#if !defined(_WIN32)
        usleep(ri == 0 ? 500000 : 200000);
#endif
    }

    for (const auto & stage : session.pipeline) {
        const dist_node_info * node = find_node(node_map, stage.node_id);
        if (node == nullptr) {
            err = "pipeline recovery node vanished while waiting: " + stage.node_id;
            return false;
        }
        if (!wait_node_worker_ready(*node, stage.role, err, 300000)) {
            err = "pipeline recovery readiness " + stage.node_id + ": " + err;
            return false;
        }
    }

    session.configure_count += 1;
    return true;
}

static bool perf_begin_decode_on_node(
        const dist_pipeline_stage & stage,
        const std::string & trace_id) {
    if (stage.http_port <= 0 || stage.host.empty() || trace_id.empty()) {
        return false;
    }
    httplib::Client cli(stage.host.c_str(), stage.http_port);
    cli.set_connection_timeout(5, 0);
    cli.set_read_timeout(15, 0);
    const json req_body = {
        { "trace_id", trace_id },
        { "perf_trace", true },
    };
    const auto res = cli.Post("/perf/trace/begin_decode", req_body.dump(), "application/json");
    return res && res->status == 200;
}

static void perf_fanout_decode_context(
        const dist_session & session,
        const std::string & trace_id) {
    if (!perf_trace_enabled() || trace_id.empty() || session.pipeline.empty()) {
        return;
    }
    std::set<std::string> seen;
    for (const auto & stage : session.pipeline) {
        if (stage.http_port <= 0 || stage.host.empty()) {
            continue;
        }
        const std::string key = stage.node_id + "@" + stage.host + ":" + std::to_string(stage.http_port);
        if (!seen.insert(key).second) {
            continue;
        }
        perf_begin_decode_on_node(stage, trace_id);
    }
}

static bool run_generation(
        const dist_session & session,
        const std::string & prompt,
        int max_new,
        json & out_tokens,
        std::string & text_out,
        std::string & err,
        json * timing_out = nullptr,
        bool apply_chat_template = false,
        const json * messages = nullptr) {
    if (session.pipeline.empty()) {
        err = "empty pipeline";
        return false;
    }

    static std::atomic<int> g_perf_trace_seq{0};
    std::string trace_id;
    const bool perf_on = perf_trace_enabled();
    bool ttft_handed_off = false;
    if (perf_on) {
        perf_trace_set_component("orchestrator");
        perf_trace_set_node_id("orchestrator");
        trace_id = perf_make_trace_id(g_perf_trace_seq.fetch_add(1) + 1);
        perf_trace_begin_ttft(trace_id, "ttft");
    }
    struct orch_ttft_cleanup {
        bool active = false;
        bool *handed_off = nullptr;
        ~orch_ttft_cleanup() {
            if (active && handed_off != nullptr && !*handed_off) {
                perf_trace_end_ttft();
            }
        }
    } ttft_cleanup{perf_on, &ttft_handed_off};

    const auto & entry = session.pipeline.front();
    if (entry.http_port <= 0) {
        err = "entry node missing http_port";
        return false;
    }

    json prompt_tokens_json = json::array();
    const runtime_role_assignment * tok_assign = session.runtime.find_role(runtime_role::tokenizer);
    if (tok_assign != nullptr && tok_assign->http_port > 0) {
        perf_ttft_span tokenize_span("TTFT_TOKENIZE", "orchestrator");
        httplib::Client tcli(tok_assign->host.c_str(), tok_assign->http_port);
        tcli.set_connection_timeout(10, 0);
        tcli.set_read_timeout(60, 0);
        json tok_req = { { "prompt", prompt }, { "chat", apply_chat_template } };
        if (messages != nullptr && messages->is_array() && !messages->empty()) {
            tok_req["messages"] = *messages;
        }
        const auto tok_res = tcli.Post("/runtime/tokenizer/tokenize", tok_req.dump(), "application/json");
        if (!tok_res || tok_res->status != 200) {
            err = "tokenizer service failed on " + tok_assign->node_id;
            return false;
        }
        try {
            const json tok_body = json::parse(tok_res->body);
            if (!tok_body.value("ok", false)) {
                err = tok_body.value("error", "tokenize failed");
                return false;
            }
            prompt_tokens_json = tok_body.at("tokens");
        } catch (...) {
            err = "invalid tokenizer response";
            return false;
        }
    }

    json body = {
        { "max_tokens", max_new },
        { "layer_end", session.entry_layer_end },
    };
    if (!prompt_tokens_json.empty()) {
        body["prompt_tokens"] = prompt_tokens_json;
    } else {
        body["prompt"] = prompt;
    }
    if (perf_on) {
        body["trace_id"]   = trace_id;
        body["perf_trace"] = true;
    }

    ttft_handed_off = true;

    if (perf_on) {
        perf_fanout_decode_context(session, trace_id);
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
        out_tokens = j.value("tokens", json::array());
        text_out   = j.value("text", "");
        if (text_out.empty() && tok_assign != nullptr && tok_assign->http_port > 0 && out_tokens.is_array()) {
            httplib::Client tcli(tok_assign->host.c_str(), tok_assign->http_port);
            tcli.set_connection_timeout(10, 0);
            tcli.set_read_timeout(60, 0);
            const json det_req = { { "tokens", out_tokens } };
            if (const auto det_res = tcli.Post("/runtime/tokenizer/detokenize", det_req.dump(), "application/json")) {
                if (det_res->status == 200) {
                    try {
                        const json det_body = json::parse(det_res->body);
                        if (det_body.value("ok", false)) {
                            text_out = det_body.value("text", "");
                        }
                    } catch (...) {}
                }
            }
        }
        if (timing_out) {
            if (j.contains("timing") && j["timing"].is_object()) {
                *timing_out = j["timing"];
            } else {
                *timing_out = json::object();
            }
            if (perf_on) {
                (*timing_out)["trace_id"] = trace_id;
                if (const char * dir = perf_trace_output_dir()) {
                    (*timing_out)["perf_trace_dir"] = dir;
                }
            }
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

// Core of session creation, extracted so the OpenAI-compatible endpoint can
// create sessions without re-entering the HTTP server (a self-call would
// occupy a second pool thread for the whole multi-minute create).
// Returns the HTTP status; `out` is the full response body in every case,
// success or failure, so the wire format is unchanged.
static int session_create_core(const json & body, json & out) {

        const std::string model_id = body.value("model", "");
        const int request_ctx = body.value("n_ctx", g_n_ctx);
        // Optional client-supplied id so a slow create can be watched through
        // GET /session/progress/{id}. Absent -> no tracking, exactly as before.
        const std::string progress_id = body.value("progress_id", "");
        progress_begin(progress_id);
        // Every early return below is a failure the watcher has to observe,
        // otherwise the UI would poll a stuck "in progress" forever.
        struct progress_fail_guard {
            std::string id;
            bool handled = false;
            ~progress_fail_guard() {
                if (!handled) {
                    progress_fail(id, "session create failed");
                }
            }
        } progress_guard{progress_id};

        // Workers are spawned during this call (setup_runtime_graph ->
        // configure_node -> perf_attach_trace), and DIST_PERF_TRACE is only
        // readable by a worker once, at its own process start -- a later
        // /session/generate enabling tracing can never reach an
        // already-spawned worker. Without this, whether a worker gets
        // traced depends on whichever stale enabled-state this orchestrator
        // process happened to be sitting in, not on this request.
        if (body.value("perf_trace", false) && !perf_trace_enabled()) {
            dist_set_env("DIST_PERF_TRACE", "1");
            perf_trace_reload_config();
        }

        if (model_id.empty()) {
            out = json({ { "error", "model id required" } });
            return 400;
        }

        dist_rss_scope rss("session_create", model_id.c_str());

        const dist_model_record * record = g_registry.find(model_id);
        if (!record) {
            out = json({ { "error", "model not registered" } });
            return 404;
        }

        // Never build a context larger than the model was trained for: the
        // extra KV would be allocated and paid for but could never be filled
        // with anything the model handles correctly.
        const int model_native_ctx = record->manifest.has_value() && record->manifest->n_ctx > 0
                ? static_cast<int>(record->manifest->n_ctx)
                : 0;
        const int effective_ctx = model_native_ctx > 0
                ? std::min(request_ctx, model_native_ctx)
                : request_ctx;

        std::map<std::string, dist_node_info> node_map;
        int n_layers = 0;
        if (record->manifest.has_value() && record->manifest->n_layer > 0) {
            n_layers = static_cast<int>(record->manifest->n_layer);
        }
        {
            std::lock_guard<std::mutex> lock(g_mu);
            for (const auto & kv : g_nodes) {
                if (node_is_online(kv.second)) {
                    node_map[kv.first] = kv.second;
                    if (n_layers <= 0 && kv.second.n_layer > 0) {
                        n_layers = std::max(n_layers, kv.second.n_layer);
                    }
                }
            }
        }

        if (node_map.size() < 1) {
            out = json({ { "error", "need at least 1 registered node" } });
            return 503;
        }

        const model_memory_requirements mem = get_model_memory_for_record(*record, effective_ctx);
        if (!mem.valid()) {
            out = json({ { "error", "failed to estimate model memory" } });
            return 503;
        }

        std::vector<dist_node_info> node_vec;
        node_vec.reserve(node_map.size());
        for (const auto & kv : node_map) {
            node_vec.push_back(kv.second);
        }
        const cluster_memory_fits_result fit = dist_check_cluster_memory_fit(mem, node_vec);

        if (!fit.fits) {
            out = fit.to_json();
            out["error"] = "model does not fit in cluster memory";
            return 503;
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
        {
            perf_session_span layout_span("SESSION_RESOLVE_LAYOUT", "orchestrator", "orchestrator");
            if (record->layout.has_value() &&
                    layout_has_full_coverage(record->layout->desired, n_layers)) {
                assignments = assignments_from_desired_layout(
                        record->layout->desired, node_map, n_layers);
            }

            if (assignments.empty()) {
                const dist_planner_result plan = dist_plan_layers_memory_aware(mem, planner_nodes);
                if (!plan.success) {
                    out = fit.to_json();
                    out["error"] = plan.error;
                    out["planning_error"] = true;
                    return 503;
                }
                assignments = plan.assignments;
            }
        }

        dist_print_planner_report(record->model_id, mem, node_vec, fit, assignments);

        const json planned = planned_layout_json(assignments);

        for (const auto & kv : node_map) {
            std::string herr;
            if (!check_node_health(kv.second, herr)) {
                out = json({
                    { "error", herr },
                    { "layout", planned },
                });
                return 503;
            }
        }

        // Self-heal a structurally unusable stored layout before anything acts
        // on it. A layout that puts every layer on one node yields a one-stage
        // pipeline, which this runtime cannot build (no final role) -- and
        // since /models/<id>/layout serves the cached copy, it would never fix
        // itself. Recomputing is cheap and bounded, so it is done inline.
        //
        // Deliberately limited to the STRUCTURALLY BROKEN case, not "the
        // layout looks suboptimal": recomputing changes where layers are
        // expected to live, which can strand already-installed blobs. The
        // coverage gate immediately below is what then reports that honestly.
        // Repairing the data itself (re-syncing gigabytes between nodes) is
        // never started from here -- a session-create request is not consent
        // to a multi-minute download.
        if (record->layout.has_value() && record->manifest.has_value() &&
                !record->layout->desired.placements.empty()) {
            std::set<std::string> layout_nodes;
            for (const auto & p : record->layout->desired.placements) {
                layout_nodes.insert(p.node_id);
            }
            if (layout_nodes.size() < 2 && node_map.size() >= 2) {
                fprintf(stderr,
                        "orchestrator: stored layout for %s uses %zu node(s) but %zu are online "
                        "-- recomputing before session setup\n",
                        model_id.c_str(), layout_nodes.size(), node_map.size());
                progress_set(progress_id, "relayout");

                std::vector<layout_node_input> relayout_nodes;
                for (const auto & kv : node_map) {
                    relayout_nodes.push_back(layout_node_from_dist(kv.second));
                }
                const auto rebuilt = build_desired_layout(
                        model_id, *record->manifest, relayout_nodes, effective_ctx, nullptr);
                if (rebuilt.success && rebuilt.layout.fits_cluster &&
                        g_registry.apply_layout(model_id, rebuilt.layout, nullptr)) {
                    store_runtime_plan_for_layout(model_id, rebuilt.layout, node_map);
                    sync_model_state_from_cluster(model_id, false);
                    persist_registry();
                    record = g_registry.find(model_id);
                    if (record == nullptr) {
                        out = json({ { "error", "model vanished during relayout" } });
                        return 500;
                    }
                    fprintf(stderr, "orchestrator: layout for %s recomputed\n", model_id.c_str());
                } else {
                    fprintf(stderr, "orchestrator: relayout of %s failed: %s\n",
                            model_id.c_str(), rebuilt.error.c_str());
                }
            }
        }

        progress_set(progress_id, "coverage_check");
        // Gate session on runtime blob coverage matching the stored install plan.
        if (record->layout.has_value() && record->manifest.has_value()) {
            perf_session_span coverage_span("SESSION_COVERAGE_CHECK", "orchestrator", "orchestrator");
            actual_model_layout actual;
            std::set<std::string> online_nodes;
            poll_installed_layers_from_nodes(model_id, actual, online_nodes);

            const semantic_runtime_descriptor rt =
                    build_semantic_runtime_descriptor(*record->manifest);
            const std::optional<runtime_install_node_map> runtime_nodes =
                    runtime_install_nodes_for_layout(*record, record->layout->desired, node_map);
            const runtime_install_node_map * runtime_ptr =
                    runtime_nodes.has_value() ? &*runtime_nodes : nullptr;
            const runtime_coverage_report rt_cov = compute_runtime_coverage(
                    rt, record->layout->desired, actual, online_nodes, runtime_ptr);

            if (!rt_cov.fully_ready()) {
                out = rt_cov.to_json();
                out["error"] = "runtime coverage not ready";
                return 503;
            }
        }

        const std::string session_id = make_id("sess");
        const bool perf_on = perf_trace_enabled();
        if (perf_on) {
            perf_trace_set_component("orchestrator");
            perf_trace_set_node_id("orchestrator");
            perf_trace_begin_session(perf_make_session_trace_id(session_id), "session");
        }
        struct session_trace_end_guard {
            bool active = false;
            ~session_trace_end_guard() {
                if (active) {
                    perf_trace_end_session();
                }
            }
        } session_trace_guard{perf_on};

        dist_session session{};
        session.session_id = session_id;
        session.model      = record->model_id;
        session.speculative_draft_model_url = body.value("speculative_draft_model_url", "");
        session.speculative_draft_k         = body.value("speculative_draft_k", 4);
        // Sampling settings are fixed for the session's lifetime: workers build
        // their sampler chain once at spawn, so changing these needs a new
        // session rather than a per-request override.
        session.temp           = body.value("temp", 0.0f);
        session.top_k          = body.value("top_k", 1);
        session.top_p          = body.value("top_p", 1.0f);
        session.min_p          = body.value("min_p", 0.0f);
        session.repeat_penalty = body.value("repeat_penalty", 1.0f);
        session.repeat_last_n  = body.value("repeat_last_n", 64);
        session.seed           = body.value("seed", 0xFFFFFFFFu);
        session.progress_id    = progress_id;
        session.n_ctx          = effective_ctx;

        std::string err;
        if (!setup_runtime_graph(session.session_id, n_layers, assignments, node_map, mem, session, err)) {
            progress_fail(progress_id, err);
            progress_guard.handled = true;
            out = json({
                { "error", err },
                { "layout", planned },
            });
            return 500;
        }

        {
            std::lock_guard<std::mutex> lock(g_mu);
            session.created_at_ms = (int64_t) std::chrono::duration_cast<std::chrono::milliseconds>(
                    std::chrono::system_clock::now().time_since_epoch()).count();
            g_sessions[session.session_id] = session;
        }

        progress_finish(progress_id, session.session_id);
        progress_guard.handled = true;

        fprintf(stderr, "orchestrator: session %s layout:", session.session_id.c_str());
        for (const auto & s : session.pipeline) {
            fprintf(stderr, " %s=[%d,%d)", s.node_id.c_str(), s.layer_start, s.layer_end);
        }
        fprintf(stderr, "\n");

        json response = {
            { "session_id", session.session_id },
            { "layout", layout_json(session) },
            { "pipeline", pipeline_json(session) },
            { "runtime_graph", session.runtime.to_json() },
            { "memory", {
                { "required_gb", mem.total_gb() },
                { "weights_gb", mem.weights_gb() },
                { "kv_gb", mem.kv_gb() },
                { "compute_gb", mem.compute_gb() },
                { "scratch_gb", mem.scratch_gb() },
            }},
            // What the pipeline can actually hold, so a caller can size
            // max_tokens instead of discovering the limit by hitting it.
            { "context", {
                { "n_ctx", session.n_ctx },
                { "model_native_n_ctx", model_native_ctx },
                { "requested_n_ctx", request_ctx },
            }},
        };
        out = response;
        return 200;
}

// ---------------------------------------------------------------------------
// Hugging Face browsing
//
// Lets the dashboard search for GGUF models and judge them against this
// cluster before committing to a download. The fit numbers here are ESTIMATES
// derived from the published file size: the exact requirement also needs
// n_layer/n_embd/n_ctx from the GGUF header, which the existing
// register -> discover -> manifest -> layout path reads (header only, no full
// download). The UI is expected to offer that as the precise check; nothing
// here should be presented as authoritative.
// ---------------------------------------------------------------------------

static httplib::Headers hf_headers() {
    httplib::Headers headers = {
        { "User-Agent", "distributed-llama-orchestrator/0.1" },
        { "Accept", "application/json" },
    };
    const std::string token = dist_hf_token();
    if (!token.empty()) {
        headers.emplace("Authorization", "Bearer " + token);
    }
    return headers;
}

// Sum of what the layout planner would actually be allowed to use: a GPU node
// contributes its free VRAM, a CPU-only node its free RAM -- matching
// layout_node_from_dist's budget choice.
static uint64_t cluster_free_budget_bytes() {
    uint64_t total = 0;
    std::lock_guard<std::mutex> lock(g_mu);
    for (const auto & kv : g_nodes) {
        if (!node_is_online(kv.second)) {
            continue;
        }
        total += kv.second.memory.has_gpu
                ? kv.second.memory.free_vram_bytes
                : kv.second.memory.free_ram_bytes;
    }
    return total;
}

// Coarse verdict from weight bytes alone. The real requirement adds KV,
// compute and scratch, which measured out at ~0.75 GB on a 28-layer 3B and
// ~1.7 GB on an 80-layer 70B -- i.e. it grows with layer count, not as a clean
// fraction of file size. Rather than invent a multiplier that is wrong at both
// ends, this only separates "clearly fits" from "tight" from "no".
static json hf_fit_estimate(uint64_t file_bytes) {
    const uint64_t budget = cluster_free_budget_bytes();
    const double file_gb = (double) file_bytes / (1024.0 * 1024.0 * 1024.0);
    const double budget_gb = (double) budget / (1024.0 * 1024.0 * 1024.0);

    std::string verdict = "unknown";
    if (budget > 0) {
        if (file_bytes > budget) {
            verdict = "no";
        } else if ((double) file_bytes * 1.15 > (double) budget) {
            verdict = "tight";
        } else {
            verdict = "fits";
        }
    }
    return {
        { "verdict", verdict },
        { "file_gb", file_gb },
        { "cluster_free_gb", budget_gb },
        { "estimated", true },
    };
}

// ---------------------------------------------------------------------------
// OpenAI-compatible surface
//
// The impedance mismatch this bridges: an OpenAI client names a model on every
// request and knows nothing about sessions, while this runtime needs an
// explicit create (seconds warm, minutes cold -- it spawns workers and loads
// weights) before it can generate. So sessions are cached and reused, keyed by
// model plus the sampling settings, because workers bake the sampler chain in
// at spawn time and a different temperature genuinely needs a different
// session.
//
// The whole surface is serialized on one mutex. That is not laziness: the
// runtime is single-stream by design (multi-tenant batching is Task 23, out of
// scope), so two concurrent generations would contend for the same workers
// anyway. Serializing makes the behavior predictable instead of racy.
// ---------------------------------------------------------------------------

static std::mutex g_oai_mu;

struct oai_cached_session {
    std::string session_id;
    int64_t     last_used_ms = 0;
};

static std::map<std::string, oai_cached_session> g_oai_sessions;

// Sessions hold worker processes and weights on every node, so an abandoned
// one is expensive. Reclaim after this much idle time.
static constexpr int64_t OAI_SESSION_IDLE_MS = 15 * 60 * 1000;

static std::string oai_cache_key(const json & body) {
    char buf[256];
    std::snprintf(buf, sizeof(buf), "|t=%.4f|k=%d|p=%.4f|mp=%.4f|rp=%.4f|rn=%d|s=%u",
            body.value("temp", 0.0f), body.value("top_k", 1), body.value("top_p", 1.0f),
            body.value("min_p", 0.0f), body.value("repeat_penalty", 1.0f),
            body.value("repeat_last_n", 64), body.value("seed", 0xFFFFFFFFu));
    return body.value("model", "") + buf;
}

static void oai_destroy_session(const std::string & session_id) {
    std::lock_guard<std::mutex> lock(g_mu);
    const auto it = g_sessions.find(session_id);
    if (it != g_sessions.end()) {
        stop_session_workers_async(it->second);
        g_sessions.erase(it);
    }
}

// Called with g_oai_mu held.
static void oai_evict_idle_locked(const std::string & keep_key) {
    const int64_t now = progress_now_ms();
    for (auto it = g_oai_sessions.begin(); it != g_oai_sessions.end();) {
        if (it->first == keep_key || now - it->second.last_used_ms < OAI_SESSION_IDLE_MS) {
            ++it;
            continue;
        }
        fprintf(stderr, "orchestrator: openai session %s idle, releasing\n",
                it->second.session_id.c_str());
        oai_destroy_session(it->second.session_id);
        it = g_oai_sessions.erase(it);
    }
}

// Resolves (creating if needed) the session backing this request. Returns an
// empty string and fills `err`/`status` on failure.
static std::string oai_session_for(const json & create_body, std::string & err, int & status) {
    const std::string key = oai_cache_key(create_body);
    oai_evict_idle_locked(key);

    const auto it = g_oai_sessions.find(key);
    if (it != g_oai_sessions.end()) {
        bool alive = false;
        {
            std::lock_guard<std::mutex> lock(g_mu);
            alive = g_sessions.count(it->second.session_id) > 0;
        }
        if (alive) {
            it->second.last_used_ms = progress_now_ms();
            return it->second.session_id;
        }
        // Destroyed out from under us (dashboard, restart) -- fall through.
        g_oai_sessions.erase(it);
    }

    json out;
    status = session_create_core(create_body, out);
    if (status != 200) {
        err = out.value("error", "session create failed");
        return {};
    }
    const std::string session_id = out.value("session_id", "");
    if (session_id.empty()) {
        status = 500;
        err = "session create returned no id";
        return {};
    }
    g_oai_sessions[key] = oai_cached_session{ session_id, progress_now_ms() };
    fprintf(stderr, "orchestrator: openai session %s created for %s\n",
            session_id.c_str(), key.c_str());
    return session_id;
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

    dist_rss_set_baseline(dist_process_rss_bytes());
    dist_rss_log_stage("startup");
    fprintf(stderr, "orchestrator: RSS diagnostics enabled (log prefix: orchestrator: RSS)\n");

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

    // CORS: the dashboard app (Electron renderer) fetches the orchestrator
    // directly from the browser, a different origin -- without these
    // headers the browser blocks the response.
    svr.set_post_routing_handler([](const httplib::Request &, httplib::Response & res) {
        res.set_header("Access-Control-Allow-Origin", "*");
        res.set_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS");
        res.set_header("Access-Control-Allow-Headers", "Content-Type");
    });
    svr.Options(R"(.*)", [](const httplib::Request &, httplib::Response & res) {
        res.status = 204;
    });

    svr.Get("/health", [](const httplib::Request &, httplib::Response & res) {
        res.set_content(R"({"status":"ok"})", "application/json");
    });

    svr.Get("/debug/rss", [](const httplib::Request &, httplib::Response & res) {
        const uint64_t rss = dist_process_rss_bytes();
        const int64_t delta = static_cast<int64_t>(rss) - static_cast<int64_t>(dist_rss_baseline_bytes());
        res.set_content(json({
            { "rss_bytes", rss },
            { "rss_mb", rss / (1024.0 * 1024.0) },
            { "baseline_bytes", dist_rss_baseline_bytes() },
            { "baseline_delta_bytes", delta },
            { "baseline_delta_mb", static_cast<double>(delta) / (1024.0 * 1024.0) },
        }).dump(), "application/json");
    });

    svr.Get("/perf/trace/list", [](const httplib::Request &, httplib::Response & res) {
        const std::string root = perf_trace_resolve_root(g_models_dir);
        res.set_content(json({
            { "node_id", "orchestrator" },
            { "root", root },
            { "files", perf_trace_list_files(root) },
        }).dump(), "application/json");
    });

    svr.Get("/perf/trace/file", [](const httplib::Request & req, httplib::Response & res) {
        const std::string rel = req.get_param_value("rel");
        const std::string root = perf_trace_resolve_root(g_models_dir);
        std::string content;
        if (!perf_trace_read_file(root, rel, content)) {
            res.status = 404;
            res.set_content(R"({"error":"not found"})", "application/json");
            return;
        }
        res.set_header("Content-Type", "application/x-ndjson");
        res.set_content(content, "application/x-ndjson");
    });

    svr.Post("/perf/trace/cleanup", [](const httplib::Request & req, httplib::Response & res) {
        int max_age_days = 7;
        try {
            if (!req.body.empty()) {
                const json body = json::parse(req.body);
                max_age_days = body.value("max_age_days", max_age_days);
            }
        } catch (...) {
            // Fall through with the default.
        }
        const std::string root = perf_trace_resolve_root(g_models_dir);
        const perf_trace_cleanup_result r = perf_trace_cleanup(root, max_age_days);
        res.set_content(json({
            { "ok", true },
            { "node_id", "orchestrator" },
            { "root", root },
            { "max_age_days", max_age_days },
            { "deleted_files", r.deleted_files },
            { "freed_bytes", r.freed_bytes },
        }).dump(), "application/json");
    });

    svr.Get("/debug/log", [](const httplib::Request & req, httplib::Response & res) {
        const std::string path = node_log_resolve_path(g_models_dir, "orchestrator.log");
        if (path.empty() || !std::filesystem::exists(path)) {
            res.status = 404;
            res.set_content(json({ { "error", "log file not found" }, { "path", path } }).dump(),
                    "application/json");
            return;
        }
        size_t lines = 300;
        if (const std::string v = req.get_param_value("lines"); !v.empty()) {
            lines = (size_t) std::max(0, std::atoi(v.c_str()));
        }
        res.set_header("Content-Type", "text/plain");
        res.set_content(node_log_tail(path, lines, 4 * 1024 * 1024), "text/plain");
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
            } else if (!node_is_online(prev_it->second)) {
                // A node that had gone quiet and is heartbeating again is a
                // membership change, even though its map entry never went
                // away and its score is unchanged.
                //
                // Nothing erases entries from g_nodes, so before this the
                // only "new node" was one that had never registered since
                // the orchestrator started. A returning machine looked like
                // an ordinary heartbeat: no resync, no re-poll, and every
                // model on it kept whatever coverage was recorded while it
                // was absent. Measured 2026-07-28 -- ten models left
                // DEGRADED, node-a back and healthy for minutes, not one of
                // them recovered until each was refreshed by hand.
                //
                // Re-polling here is safe to do eagerly: what may actually
                // be *done* to a layout is decided by decide_layout_action(),
                // which keeps or repairs and only recomputes when there is no
                // runnable layout to protect.
                cluster_changed = true;
                fprintf(stderr, "orchestrator: %s is back after being offline -- resyncing\n",
                        node.node_id.c_str());
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
                    { "online", node_is_online(n) },
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
                if (!node_is_online(kv.second)) {
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
        json out;
        res.status = session_create_core(body, out);
        res.set_content(out.dump(), "application/json");
    });

    // GET /hf/search?q=...&limit=  -- browse GGUF repositories on Hugging Face.
    svr.Get("/hf/search", [](const httplib::Request & req, httplib::Response & res) {
        const std::string q = req.get_param_value("q");
        const std::string limit = req.has_param("limit") ? req.get_param_value("limit") : "24";

        httplib::Client cli("https://huggingface.co");
        cli.set_connection_timeout(10, 0);
        cli.set_read_timeout(30, 0);
        cli.set_follow_location(true);

        std::string path = "/api/models?filter=gguf&sort=downloads&direction=-1&limit=" + limit;
        if (!q.empty()) {
            path += "&search=" + httplib::encode_uri_component(q);
        }

        const auto hf = cli.Get(path.c_str(), hf_headers());
        if (!hf) {
            res.status = 502;
            res.set_content(json({ { "error", "could not reach Hugging Face" } }).dump(), "application/json");
            return;
        }
        if (hf->status != 200) {
            res.status = hf->status;
            res.set_content(json({
                { "error", "Hugging Face API returned HTTP " + std::to_string(hf->status) },
            }).dump(), "application/json");
            return;
        }

        json parsed;
        try {
            parsed = json::parse(hf->body);
        } catch (...) {
            res.status = 502;
            res.set_content(json({ { "error", "invalid JSON from Hugging Face" } }).dump(), "application/json");
            return;
        }

        json models = json::array();
        if (parsed.is_array()) {
            for (const auto & m : parsed) {
                models.push_back({
                    { "repository", m.value("id", "") },
                    { "downloads", m.value("downloads", 0) },
                    { "likes", m.value("likes", 0) },
                });
            }
        }
        res.set_content(json({ { "models", models } }).dump(), "application/json");
    });

    // GET /hf/files?repo=...  -- GGUF files in a repo, with a size-based fit
    // estimate against this cluster. Estimates only; see hf_fit_estimate.
    svr.Get("/hf/files", [](const httplib::Request & req, httplib::Response & res) {
        const std::string repo = req.get_param_value("repo");
        if (repo.empty()) {
            res.status = 400;
            res.set_content(json({ { "error", "repo required" } }).dump(), "application/json");
            return;
        }

        httplib::Client cli("https://huggingface.co");
        cli.set_connection_timeout(10, 0);
        cli.set_read_timeout(30, 0);
        cli.set_follow_location(true);

        const auto hf = cli.Get(("/api/models/" + repo + "/tree/main").c_str(), hf_headers());
        if (!hf) {
            res.status = 502;
            res.set_content(json({ { "error", "could not reach Hugging Face" } }).dump(), "application/json");
            return;
        }
        if (hf->status != 200) {
            res.status = hf->status;
            res.set_content(json({
                { "error", "Hugging Face API returned HTTP " + std::to_string(hf->status) },
            }).dump(), "application/json");
            return;
        }

        json parsed;
        try {
            parsed = json::parse(hf->body);
        } catch (...) {
            res.status = 502;
            res.set_content(json({ { "error", "invalid JSON from Hugging Face" } }).dump(), "application/json");
            return;
        }

        json files = json::array();
        if (parsed.is_array()) {
            for (const auto & f : parsed) {
                const std::string path = f.value("path", "");
                if (path.size() < 5 || path.compare(path.size() - 5, 5, ".gguf") != 0) {
                    continue;
                }
                const uint64_t size = f.contains("size") && f["size"].is_number()
                        ? f["size"].get<uint64_t>() : 0;
                files.push_back({
                    { "filename", path },
                    { "size_bytes", size },
                    { "fit", hf_fit_estimate(size) },
                });
            }
        }
        res.set_content(json({
            { "repository", repo },
            { "files", files },
            { "cluster_free_gb", (double) cluster_free_budget_bytes() / (1024.0 * 1024.0 * 1024.0) },
        }).dump(), "application/json");
    });

    // GET /hf/draft-candidates?repo=...  -- look for a speculative draft model
    // to pair with a target.
    //
    // There is no declared field for this. Hugging Face metadata has no
    // draft/speculative key, `base_model` points at a GGUF's unquantized
    // source rather than a pair, and target model cards do not mention drafts
    // at all (checked on Llama-3.3-70B: zero occurrences).
    //
    // What does exist: draft models name their target in the repo name
    // (alamios/Mistral-Small-3.1-DRAFT-0.5B) and state the pairing in README
    // prose ("meant to be used as draft model for speculative decoding with
    // <link>"). So this searches by family and reports the evidence it found
    // per candidate -- it is a shortlist to measure, never an assertion that a
    // pair works. Acceptance is strongly pair-dependent: the same 1B draft gave
    // x1.64 on Llama-3.2-3B and a 19% hit rate with no speedup on
    // Llama-3.3-70B.
    svr.Get("/hf/draft-candidates", [](const httplib::Request & req, httplib::Response & res) {
        const std::string repo = req.get_param_value("repo");
        if (repo.empty()) {
            res.status = 400;
            res.set_content(json({ { "error", "repo required" } }).dump(), "application/json");
            return;
        }

        // Family stem: drop the owner, then the quant/format suffixes that
        // would never appear in a draft's name ("bartowski/Qwen3-14B-GGUF" ->
        // "Qwen3").
        std::string stem = repo.substr(repo.find('/') + 1);
        for (const char * suffix : { "-GGUF", "-gguf", "-Instruct", "-instruct" }) {
            const size_t at = stem.find(suffix);
            if (at != std::string::npos) {
                stem = stem.substr(0, at);
            }
        }
        // Keep the family and its size marker, e.g. "Qwen3-14B" -> "Qwen3".
        const size_t dash = stem.find('-');
        const std::string family = dash == std::string::npos ? stem : stem.substr(0, dash);

        // The target's own size marker ("70b", "14b"). A candidate carrying it
        // is the target itself under some other name, not a draft for it --
        // searching "Llama draft" turns up
        // Llama-3.3-70B-Instruct-Q3_K_M-draft-layers, which is the 70B.
        std::string target_size;
        {
            std::string lower_stem = stem;
            std::transform(lower_stem.begin(), lower_stem.end(), lower_stem.begin(),
                    [](unsigned char c) { return (char) std::tolower(c); });
            for (size_t i = 0; i + 1 < lower_stem.size(); ++i) {
                if (!std::isdigit((unsigned char) lower_stem[i])) {
                    continue;
                }
                size_t j = i;
                while (j < lower_stem.size() && std::isdigit((unsigned char) lower_stem[j])) {
                    ++j;
                }
                if (j < lower_stem.size() && lower_stem[j] == 'b') {
                    target_size = lower_stem.substr(i, j - i + 1);
                }
            }
        }

        httplib::Client cli("https://huggingface.co");
        cli.set_connection_timeout(10, 0);
        cli.set_read_timeout(30, 0);
        cli.set_follow_location(true);

        const std::string query = family + " draft";
        const std::string path = "/api/models?filter=gguf&sort=downloads&direction=-1&limit=10&search=" +
                httplib::encode_uri_component(query);

        const auto hf = cli.Get(path.c_str(), hf_headers());
        if (!hf || hf->status != 200) {
            res.status = hf ? hf->status : 502;
            res.set_content(json({ { "error", "Hugging Face search failed" } }).dump(), "application/json");
            return;
        }

        json parsed;
        try {
            parsed = json::parse(hf->body);
        } catch (...) {
            res.status = 502;
            res.set_content(json({ { "error", "invalid JSON from Hugging Face" } }).dump(), "application/json");
            return;
        }

        json candidates = json::array();
        if (parsed.is_array()) {
            for (const auto & m : parsed) {
                const std::string id = m.value("id", "");
                if (id.empty() || id == repo) {
                    continue;
                }
                // Only things that actually call themselves a draft.
                std::string lower = id;
                std::transform(lower.begin(), lower.end(), lower.begin(),
                        [](unsigned char c) { return (char) std::tolower(c); });
                if (lower.find("draft") == std::string::npos &&
                        lower.find("drafter") == std::string::npos) {
                    continue;
                }
                // Same size as the target means it is the target, not a draft.
                if (!target_size.empty() && lower.find(target_size) != std::string::npos) {
                    continue;
                }
                candidates.push_back({
                    { "repository", id },
                    { "downloads", m.value("downloads", 0) },
                    // Named after the family we searched for; the pairing is
                    // still unproven until measured.
                    { "evidence", "name matches family + draft" },
                });
            }
        }

        res.set_content(json({
            { "target", repo },
            { "family", family },
            { "candidates", candidates },
            // Said explicitly so the UI cannot present this as a fact.
            { "verified", false },
            { "note", "heuristic shortlist; Hugging Face has no declared draft-pair "
                      "field. Acceptance must be measured -- it is strongly "
                      "pair-dependent." },
        }).dump(), "application/json");
    });

    // GET /v1/models -- OpenAI shape. Only models that are actually installed
    // are listed; a client picking one from here should be able to use it.
    svr.Get("/v1/models", [](const httplib::Request &, httplib::Response & res) {
        json data = json::array();
        for (const auto & m : g_registry.list()) {
            const bool ready = m.coverage.has_value() &&
                    m.coverage->state == coverage_state::ready;
            if (!ready) {
                continue;
            }
            data.push_back({
                { "id", m.model_id },
                { "object", "model" },
                { "created", 0 },
                { "owned_by", "distributed-llm" },
            });
        }
        res.set_content(json({ { "object", "list" }, { "data", data } }).dump(), "application/json");
    });

    // POST /v1/chat/completions -- OpenAI shape, non-streaming.
    svr.Post("/v1/chat/completions", [](const httplib::Request & req, httplib::Response & res) {
        const auto fail = [&res](int status, const std::string & message, const std::string & type) {
            res.status = status;
            res.set_content(json({ { "error", {
                { "message", message }, { "type", type }, { "code", nullptr },
            } } }).dump(), "application/json");
        };

        json body;
        try {
            body = json::parse(req.body);
        } catch (...) {
            fail(400, "invalid json", "invalid_request_error");
            return;
        }

        const std::string model = body.value("model", "");
        if (model.empty()) {
            fail(400, "model required", "invalid_request_error");
            return;
        }
        if (!body.contains("messages") || !body["messages"].is_array() || body["messages"].empty()) {
            fail(400, "messages required", "invalid_request_error");
            return;
        }
        if (body.value("stream", false)) {
            // Better an explicit refusal than silently returning a
            // non-streaming body a streaming client will fail to parse.
            fail(400, "streaming is not supported yet; set stream=false", "invalid_request_error");
            return;
        }

        // OpenAI names only a subset of what the sampler chain takes; the rest
        // stay at this runtime's defaults.
        json create_body = { { "model", model } };
        if (body.contains("temperature")) create_body["temp"] = body["temperature"].get<float>();
        if (body.contains("top_p"))       create_body["top_p"] = body["top_p"].get<float>();
        if (body.contains("seed"))        create_body["seed"] = body["seed"].get<unsigned int>();
        if (body.contains("frequency_penalty")) {
            create_body["repeat_penalty"] = 1.0f + body["frequency_penalty"].get<float>();
        }
        // A temperature above zero has to leave the greedy path, otherwise the
        // request would silently sample deterministically anyway.
        if (create_body.value("temp", 0.0f) > 0.0f && !create_body.contains("top_k")) {
            create_body["top_k"] = 0;   // 0 -> disabled, consider the whole vocab
        }

        std::lock_guard<std::mutex> oai_lock(g_oai_mu);

        std::string err;
        int status = 200;
        const std::string session_id = oai_session_for(create_body, err, status);
        if (session_id.empty()) {
            const int code = status == 200 ? 500 : status;
            // Clients branch on this: a 404 for an unknown model is the
            // caller's problem, a 503 from the cluster is ours.
            fail(code, err, code < 500 ? "invalid_request_error" : "server_error");
            return;
        }

        dist_session session{};
        {
            std::lock_guard<std::mutex> lock(g_mu);
            const auto it = g_sessions.find(session_id);
            if (it == g_sessions.end()) {
                fail(500, "session vanished", "server_error");
                return;
            }
            session = it->second;
        }

        const int max_tokens = body.value("max_tokens", DIST_MAX_NEW_TOKENS);
        const json messages  = body["messages"];

        json out_tokens = json::array();
        std::string text;
        json timing = json::object();
        std::string gen_err;
        if (!run_generation(session, "", max_tokens, out_tokens, text, gen_err,
                    &timing, true, &messages)) {
            fail(500, gen_err, "server_error");
            return;
        }

        const int64_t now_s = progress_now_ms() / 1000;
        const int completion_tokens = (int) out_tokens.size();
        res.set_content(json({
            { "id", "chatcmpl-" + session_id },
            { "object", "chat.completion" },
            { "created", now_s },
            { "model", model },
            { "choices", json::array({ json{
                { "index", 0 },
                { "message", { { "role", "assistant" }, { "content", text } } },
                // Nothing distinguishes "hit the token cap" from "the model
                // stopped" at this layer yet, so report the honest generic value.
                { "finish_reason", completion_tokens >= max_tokens ? "length" : "stop" },
            } }) },
            { "usage", {
                { "prompt_tokens", 0 },
                { "completion_tokens", completion_tokens },
                { "total_tokens", completion_tokens },
            } },
        }).dump(), "application/json");
    });

    // GET /session/progress/{id} -- poll a create in flight. `id` is whatever
    // the client passed as progress_id on /session/create.
    svr.Get(R"(/session/progress/([^/]+))", [](const httplib::Request & req, httplib::Response & res) {
        const std::string id = req.matches[1];
        std::lock_guard<std::mutex> lock(g_progress_mu);
        const auto it = g_session_progress.find(id);
        if (it == g_session_progress.end()) {
            res.status = 404;
            res.set_content(json({ { "error", "unknown progress id" } }).dump(), "application/json");
            return;
        }
        res.set_content(it->second.to_json().dump(), "application/json");
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
                if (node_is_online(kv.second) && kv.second.n_layer > 0) {
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
                if (node_is_online(kv.second) && kv.second.n_layer > 0) {
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
        const std::string prompt     = body.value("prompt", DEFAULT_GENERATE_PROMPT);
        const int max_tokens         = body.value("max_tokens", DIST_MAX_NEW_TOKENS);
        const bool chat_mode         = body.value("chat", false);
        // messages[] carries a full conversation; the legacy single `prompt`
        // string still works and is what the non-chat callers (ceiling
        // measurement, soak scripts) send.
        const json messages          = body.contains("messages") && body["messages"].is_array()
                ? body["messages"] : json::array();
        const json * messages_ptr    = messages.empty() ? nullptr : &messages;

        // Request-driven trace enablement: the benchmark harness runs on a
        // different host, so gating fanout on this process's env alone means
        // silent no-trace runs unless the orchestrator was started traced.
        if (body.value("perf_trace", false) && !perf_trace_enabled()) {
            dist_set_env("DIST_PERF_TRACE", "1");
            perf_trace_reload_config();
        }

        const auto t_request_start = std::chrono::steady_clock::now();

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

        dist_rss_scope rss("generate", session.model.c_str());

        json out_tokens = json::array();
        std::string text;
        std::string err;
        json pipeline_timing = json::object();
        bool ok = run_generation(session, prompt, max_tokens, out_tokens, text, err, &pipeline_timing, chat_mode, messages_ptr);
        if (!ok) {
            const std::string first_err = err;
            std::string recovery_err;
            dist_session recovered = session;
            if (recover_session_pipeline(recovered, recovery_err)) {
                json retry_timing = json::object();
                std::string retry_err;
                json retry_tokens = json::array();
                std::string retry_text;
                if (run_generation(recovered, prompt, max_tokens, retry_tokens, retry_text, retry_err, &retry_timing, chat_mode, messages_ptr)) {
                    ok = true;
                    out_tokens = std::move(retry_tokens);
                    text = std::move(retry_text);
                    pipeline_timing = std::move(retry_timing);
                    pipeline_timing["recovered_pipeline"] = true;
                    pipeline_timing["first_error"] = first_err;
                    {
                        std::lock_guard<std::mutex> lock(g_mu);
                        auto it = g_sessions.find(session_id);
                        if (it != g_sessions.end()) {
                            it->second = recovered;
                        }
                    }
                } else {
                    err = first_err + "; retry after pipeline recovery failed: " + retry_err;
                    pipeline_timing["recovered_pipeline"] = false;
                    pipeline_timing["recovery_attempted"] = true;
                    pipeline_timing["recovery_error"] = retry_err;
                }
            } else {
                err = first_err + "; pipeline recovery failed: " + recovery_err;
                pipeline_timing["recovered_pipeline"] = false;
                pipeline_timing["recovery_attempted"] = true;
                pipeline_timing["recovery_error"] = recovery_err;
            }
        }

        {
            std::lock_guard<std::mutex> lock(g_mu);
            auto it = g_sessions.find(session_id);
            if (it != g_sessions.end()) {
                it->second.generate_count++;
            }
        }

        const auto t_request_end = std::chrono::steady_clock::now();
        const double total_ms = std::chrono::duration<double, std::milli>(t_request_end - t_request_start).count();
        const int generated_n = out_tokens.is_array() ? (int) out_tokens.size() : 0;

        json timing = pipeline_timing;
        if (!timing.contains("total_ms")) {
            timing["total_ms"] = total_ms;
        }
        if (!timing.contains("ttft_ms") && timing.contains("prefill_ms")) {
            timing["ttft_ms"] = timing["prefill_ms"];
        }
        if (!timing.contains("decode_ms") && generated_n > 1 && timing.contains("total_ms")) {
            const double ttft = timing.value("ttft_ms", timing.value("prefill_ms", 0.0));
            timing["decode_ms"] = std::max(0.0, timing["total_ms"].get<double>() - ttft);
        }
        if (generated_n > 0 && timing.contains("decode_ms")) {
            const double decode_ms = timing["decode_ms"].get<double>();
            const int decode_tokens = std::max(1, generated_n - 1);
            timing["decode_tokens_per_sec"] = decode_ms > 0 ? (decode_tokens * 1000.0 / decode_ms) : 0.0;
        }
        timing["generated_tokens"] = generated_n;

        if (!ok) {
            res.status = 503;
            res.set_content(json({
                { "error", err },
                { "tokens", out_tokens },
                { "text", text },
                { "timing", timing },
            }).dump(), "application/json");
            return;
        }

        res.set_content(json({
            { "session_id", session_id },
            { "tokens", out_tokens },
            { "text", text },
            { "count", generated_n },
            { "timing", timing },
            { "session_stats", {
                { "generate_count", session.generate_count + 1 },
            }},
        }).dump(), "application/json");
    });

    svr.Get(R"(/session/([^/]+))", [](const httplib::Request & req, httplib::Response & res) {
        const std::string session_id = req.matches[1];
        std::lock_guard<std::mutex> lock(g_mu);
        const auto it = g_sessions.find(session_id);
        if (it == g_sessions.end()) {
            res.status = 404;
            res.set_content(json({ { "error", "session not found" } }).dump(), "application/json");
            return;
        }
        res.set_content(session_debug_json(it->second).dump(), "application/json");
    });

    svr.Delete(R"(/session/([^/]+))", [](const httplib::Request & req, httplib::Response & res) {
        const std::string session_id = req.matches[1];
        bool removed = false;
        {
            std::lock_guard<std::mutex> lock(g_mu);
            const auto it = g_sessions.find(session_id);
            if (it != g_sessions.end()) {
                stop_session_workers_async(it->second);
                g_sessions.erase(it);
                removed = true;
            }
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
            const auto it = g_sessions.find(session_id);
            if (it != g_sessions.end()) {
                stop_session_workers_async(it->second);
                g_sessions.erase(it);
                removed = true;
            }
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
        dist_rss_scope rss("register", model_id.c_str());

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
                if (node_is_online(kv.second)) {
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
                if (node_is_online(node)) {
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
        dist_rss_scope rss("discover", model_id.c_str());

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
        dist_rss_scope rss("manifest", model_id.c_str());

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
        dist_rss_scope rss("layout", model_id.c_str());

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
            if (!record->stored_runtime_graph.has_value()) {
                std::map<std::string, dist_node_info> node_map;
                {
                    std::lock_guard<std::mutex> lock(g_mu);
                    for (const auto & kv : g_nodes) {
                        if (node_is_online(kv.second)) {
                            node_map[kv.first] = kv.second;
                        }
                    }
                }
                store_runtime_plan_for_layout(model_id, record->layout->desired, node_map);
                record = g_registry.find(model_id);
            }
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
                if (node_is_online(kv.second)) {
                    nodes.push_back(layout_node_from_dist(kv.second));
                }
            }
        }
        desired_model_layout previous;
        if (record->layout.has_value()) {
            previous = record->layout->desired;
        }

        // Pass the previous layout so role ordering only changes when a node
        // is meaningfully stronger, not merely momentarily luckier (score is
        // measured live and drifts under unrelated load).
        const auto built = build_desired_layout(
                model_id, *record->manifest, nodes, request_ctx,
                previous.placements.empty() ? nullptr : &previous);
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

        {
            std::map<std::string, dist_node_info> node_map;
            {
                std::lock_guard<std::mutex> lock(g_mu);
                for (const auto & kv : g_nodes) {
                    if (node_is_online(kv.second)) {
                        node_map[kv.first] = kv.second;
                    }
                }
            }
            store_runtime_plan_for_layout(model_id, built.layout, node_map);
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

        // Relaying out an UNAVAILABLE model abandons the absent node's copies
        // and re-fetches them elsewhere. Right when that machine is gone,
        // destructive when it is merely off -- so say so and let the caller
        // decide, rather than guessing on their behalf.
        {
            const unavailable_advice advice = describe_unavailable(*record);
            if (advice.applies && !body_confirms_unavailable(req.body)) {
                json out = unavailable_advice_json(advice, model_id);
                out["error"] = "model is UNAVAILABLE -- refusing to relayout without confirmation";
                res.status = 409;
                res.set_content(out.dump(), "application/json");
                return;
            }
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
        dist_rss_scope rss("coverage", model_id.c_str());

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

        if (!g_registry.apply_actual(model_id, actual, online_nodes, record)) {
            res.status = 404;
            res.set_content(json({ { "error", "model not registered" } }).dump(), "application/json");
            return;
        }
        if (!refresh_model_coverage(model_id, online_nodes, record)) {
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

        if (!g_registry.apply_actual(model_id, actual, online_nodes, record)) {
            res.status = 404;
            res.set_content(json({ { "error", "model not registered" } }).dump(), "application/json");
            return;
        }
        refresh_model_coverage(model_id, online_nodes, record);

        // Same reason as build_and_store_install_plan: the raw poll omits
        // nodes that did not answer, and reconciling against it would call
        // their intact layers missing.
        const actual_model_layout & actual_merged =
                record->actual.has_value() ? *record->actual : actual;

        const reconciliation_result result = reconcile_layers(
                record->layout->desired, actual_merged, online_nodes);
        coverage_report coverage = compute_coverage(
                record->layout->desired, actual_merged, online_nodes);
        if (record->manifest.has_value()) {
            const semantic_runtime_descriptor rt =
                    build_semantic_runtime_descriptor(*record->manifest);
            std::map<std::string, dist_node_info> node_map;
            {
                std::lock_guard<std::mutex> lock(g_mu);
                for (const std::string & nid : online_nodes) {
                    const auto it = g_nodes.find(nid);
                    if (it != g_nodes.end()) {
                        node_map[nid] = it->second;
                    }
                }
            }
            const std::optional<runtime_install_node_map> runtime_nodes =
                    runtime_install_nodes_for_layout(*record, record->layout->desired, node_map);
            const runtime_install_node_map * runtime_ptr =
                    runtime_nodes.has_value() ? &*runtime_nodes : nullptr;
            coverage = compute_runtime_coverage(
                    rt, record->layout->desired, actual_merged, online_nodes, runtime_ptr).layer_coverage;
        }
        g_registry.apply_coverage(model_id, coverage, record);

        res.set_content(result.to_json().dump(), "application/json");
    });

    // POST /models/{model_id}/install-plan - Build install plan from registry state.
    svr.Post(R"(/models/([^/]+)/install-plan)", [](const httplib::Request & req, httplib::Response & res) {
        const std::string model_id = req.matches[1];
        dist_rss_scope rss("install_plan", model_id.c_str());

        const bool perf_on = perf_trace_enabled();
        if (perf_on) {
            perf_trace_set_component("orchestrator");
            perf_trace_set_node_id("orchestrator");
            perf_trace_begin_install("install-plan-" + model_id, "install");
        }
        struct install_trace_end_guard {
            bool active = false;
            ~install_trace_end_guard() {
                if (active) {
                    perf_trace_end_install();
                }
            }
        } install_trace_guard{perf_on};
        perf_install_span plan_build("INSTALL_PLAN_BUILD", "plan");

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

        json plan_out = json{
            { "status", "ok" },
            { "model", model_id },
            { "operation_count", record->stored_install_plan->operation_count },
            { "total_download_bytes", record->stored_install_plan->total_download_bytes },
            { "install_plan", record->stored_install_plan->to_json() },
        };
        // Read-only call, so nothing to refuse -- but this is where an operator
        // looks before repairing, and it is the right moment to say that
        // turning the machine back on is free and this is not.
        {
            const unavailable_advice advice = describe_unavailable(*record);
            if (advice.applies) {
                plan_out["warning"] = unavailable_advice_json(advice, model_id);
            }
        }
        res.set_content(plan_out.dump(), "application/json");

        if (perf_on) {
            char attrs[256];
            std::snprintf(
                    attrs,
                    sizeof(attrs),
                    "{\"operation_count\":%d,\"total_download_bytes\":%llu}",
                    record->stored_install_plan->operation_count,
                    static_cast<unsigned long long>(record->stored_install_plan->total_download_bytes));
            perf_emit_install_instant(
                    "INSTALL_PLAN_READY",
                    "plan",
                    nullptr,
                    nullptr,
                    record->stored_install_plan->total_download_bytes,
                    attrs);
        }
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

        // Same gate as /optimize: this is the call that actually moves bytes.
        {
            const unavailable_advice advice = describe_unavailable(*record);
            if (advice.applies && !body_confirms_unavailable(req.body)) {
                json out = unavailable_advice_json(advice, model_id);
                out["error"] = "model is UNAVAILABLE -- refusing to install without confirmation";
                res.status = 409;
                res.set_content(out.dump(), "application/json");
                return;
            }
        }

        std::map<std::string, dist_node_info> nodes_copy;
        {
            std::lock_guard<std::mutex> lock(g_mu);
            for (const auto & kv : g_nodes) {
                if (node_is_online(kv.second)) {
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
        g_registry.apply_actual(model_id, actual, online_nodes, record);
        refresh_model_coverage(model_id, online_nodes, record);
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
                if (node_is_online(kv.second)) {
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
