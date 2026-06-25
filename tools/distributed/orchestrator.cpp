#include "dist_common.h"
#include "layer_planner.h"
#include "memory_estimator.h"
#include "model_catalog.h"
#include "split_gen_common.h"
#include "split_tcp_wire.h"

#include "httplib.h"
#include "nlohmann/json.hpp"
#include "llama.h"

#include "ggml-backend.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <map>
#include <mutex>
#include <random>
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
static int32_t g_n_ctx = 4096;
static model_catalog g_catalog;
static std::map<std::string, json> g_install_node_results;

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

static model_memory_requirements get_model_memory(
        const std::string & model_id,
        int32_t             n_ctx) {
    // Prefer the orchestrator's local GGUF if it exists.
    if (!g_model_path.empty() && std::filesystem::exists(g_model_path)) {
        model_memory_requirements mem = estimate_model_memory(g_model_path, n_ctx);
        if (mem.valid()) {
            mem.model_id = model_id;
            return mem;
        }
    }

    // Otherwise fall back to the catalog entry.
    std::lock_guard<std::mutex> lock(g_mu);
    const auto * model = g_catalog.find_model(model_id);
    if (model) {
        return estimate_model_memory_from_catalog(*model, n_ctx);
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
    const size_t n_stages = assignments.size();

    std::vector<dist_pipeline_stage> stages;
    stages.reserve(n_stages);

    for (size_t i = 0; i < n_stages; ++i) {
        const auto & assign = assignments[i];
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
        } else if (i + 1 == n_stages) {
            stage.role      = DIST_ROLE_FINAL;
            stage.peer_port = pipe_base + (int) i + 1;
        } else {
            stage.role      = DIST_ROLE_MIDDLE;
            stage.peer_port = pipe_base + (int) i + 1;
        }

        stages.push_back(stage);
    }

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
            { "layer_start", stage.layer_start },
            { "layer_end", stage.layer_end },
            { "peer_bind", "0.0.0.0" },
        };

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
            const auto & next = stages[1];
            cfg["role"] = "entry";
            cfg["ctrl_port"] = stage.ctrl_port;
            cfg["next_host"] = next.host;
            cfg["next_port"] = next.peer_port;
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
    int ctrl_fd = split_tcp_connect_retry(session.entry_host.c_str(), session.entry_ctrl_port, 300, 100);
    if (ctrl_fd < 0) {
        err = "failed to connect to entry node ctrl port";
        return false;
    }

    split_gen3_a_resp resp{};

    if (!gen3_send_recv(ctrl_fd, SPLIT_GEN_CMD_RESET, 0, 0, session.entry_layer_end, nullptr, resp)) {
        err = "reset failed";
        close(ctrl_fd);
        return false;
    }

    std::vector<int32_t> ptoks(prompt.size());
    for (size_t i = 0; i < prompt.size(); ++i) {
        ptoks[i] = (int32_t) prompt[i];
    }

    if (!gen3_send_recv(ctrl_fd, SPLIT_GEN_CMD_PREFILL, (int32_t) prompt.size(), 0,
            session.entry_layer_end, ptoks.data(), resp)) {
        err = "prefill failed";
        close(ctrl_fd);
        return false;
    }

    if (resp.token_id < 0) {
        err = "prefill returned error from pipeline";
        close(ctrl_fd);
        return false;
    }

    out_tokens.push_back((llama_token) resp.token_id);
    llama_token cur = (llama_token) resp.token_id;

    const int n_prompt = (int) prompt.size();
    for (int step = 1; step < max_new; ++step) {
        const int32_t tok_i32 = (int32_t) cur;
        const int32_t pos     = n_prompt + step - 1;

        if (!gen3_send_recv(ctrl_fd, SPLIT_GEN_CMD_DECODE, 1, pos, session.entry_layer_end, &tok_i32, resp)) {
            err = "decode failed at step " + std::to_string(step);
            close(ctrl_fd);
            return false;
        }

        if (resp.token_id < 0) {
            err = "pipeline error at step " + std::to_string(step);
            close(ctrl_fd);
            return false;
        }

        cur = (llama_token) resp.token_id;
        out_tokens.push_back(cur);
    }

    gen3_send_recv(ctrl_fd, SPLIT_GEN_CMD_SHUTDOWN, 0, 0, session.entry_layer_end, nullptr, resp);
    close(ctrl_fd);
    return true;
}

static void usage(const char * prog) {
    fprintf(stderr, "usage: %s --model PATH [--listen HOST:PORT] [--ctx-size N]\n", prog);
}

int main(int argc, char ** argv) {
    std::string listen = "0.0.0.0:9000";

    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "--listen") == 0 && i + 1 < argc) {
            listen = argv[++i];
        } else if (strcmp(argv[i], "--model") == 0 && i + 1 < argc) {
            g_model_path = argv[++i];
        } else if (strcmp(argv[i], "--ctx-size") == 0 && i + 1 < argc) {
            g_n_ctx = std::atoi(argv[++i]);
        } else {
            usage(argv[0]);
            return 1;
        }
    }

    if (g_model_path.empty()) {
        usage(argv[0]);
        return 1;
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
        node.memory_total_mb = body.value("memory_total_mb", 0);
        node.memory_free_mb  = body.value("memory_free_mb", 0);
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
            node.memory.total_ram_bytes  = mem.value("total_ram", 0);
            node.memory.free_ram_bytes   = mem.value("free_ram", 0);
            node.memory.total_vram_bytes = mem.value("total_vram", 0);
            node.memory.free_vram_bytes  = mem.value("free_vram", 0);
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
            if (caps.contains("total_ram"))  node.caps.total_ram_bytes  = caps.value("total_ram", 0);
            if (caps.contains("free_ram"))   node.caps.free_ram_bytes   = caps.value("free_ram", 0);
            if (caps.contains("total_vram")) node.caps.total_vram_bytes = caps.value("total_vram", 0);
            if (caps.contains("free_vram"))  node.caps.free_vram_bytes  = caps.value("free_vram", 0);
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

        {
            std::lock_guard<std::mutex> lock(g_mu);
            g_nodes[node.node_id] = node;
        }

        fprintf(stderr, "orchestrator: registered node %s at %s:%d layers=%d score=%.1f decode=%.1f prefill=%.1f "
                "ram=%.1fGB vram=%.1fGB has_gpu=%d\n",
                node.node_id.c_str(), node.host.c_str(), node.http_port, node.n_layer, node.score,
                node.performance.decode_tps, node.performance.prefill_tps,
                dist_bytes_to_gb(node.memory.free_ram_bytes),
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

        const std::string model = body.value("model", "llama-3.2-1b");
        const int request_ctx = body.value("n_ctx", g_n_ctx);

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

        if (node_map.size() < 1) {
            res.status = 503;
            res.set_content(json({ { "error", "need at least 1 registered node" } }).dump(), "application/json");
            return;
        }

        const model_memory_requirements mem = get_model_memory(model, request_ctx);
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

        const dist_planner_result plan = dist_plan_layers_memory_aware(mem, planner_nodes);
        if (!plan.success) {
            res.status = 503;
            json err = fit.to_json();
            err["error"] = plan.error;
            err["planning_error"] = true;
            res.set_content(err.dump(), "application/json");
            return;
        }

        dist_print_planner_report(model, mem, node_vec, fit, plan.assignments);

        const json planned = planned_layout_json(plan.assignments);

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
        session.model      = model;
        session.model_path = g_model_path;

        std::string err;
        if (!setup_pipeline(session.session_id, n_layers, plan.assignments, node_map, session, err)) {
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

        const std::string model = body.value("model", "");
        const int request_ctx = body.value("n_ctx", g_n_ctx);

        if (model.empty()) {
            res.status = 400;
            res.set_content(R"({"error":"model is required"})", "application/json");
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

        const model_memory_requirements mem = get_model_memory(model, request_ctx);
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
        response["model"] = model;
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
        const std::string model = req.get_param_value("model");
        if (model.empty()) {
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
        response["model"] = model;
        response["n_ctx"] = request_ctx;

        if (node_map.empty()) {
            res.status = 503;
            response["error"] = "no online nodes";
            res.set_content(response.dump(), "application/json");
            return;
        }

        const model_memory_requirements mem = get_model_memory(model, request_ctx);
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

        explanation << "Model " << model << " requires " << mem.total_gb()
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
        llama_model * model = llama_model_load_from_file(session.model_path.c_str(), llama_model_default_params());
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

    // GET /models - List installed models across cluster (aggregated from nodes)
    svr.Get("/models", [](const httplib::Request &, httplib::Response & res) {
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
            const auto * found = g_catalog.find_model(model_id);
            if (!found) {
                res.status = 404;
                res.set_content(R"({"error":"model not found in catalog"})", "application/json");
                return;
            }

            model = *found;
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

    fprintf(stderr, "orchestrator: listening on %s:%d (dynamic layer planner)\n", bind_host.c_str(), port);

    if (!svr.listen(bind_host.c_str(), port)) {
        fprintf(stderr, "orchestrator: failed to bind\n");
        return 1;
    }

    return 0;
}
