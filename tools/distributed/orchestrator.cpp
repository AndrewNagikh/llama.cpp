#include "dist_common.h"
#include "layer_planner.h"
#include "model_catalog.h"
#include "split_gen_common.h"
#include "split_tcp_wire.h"

#include "httplib.h"
#include "nlohmann/json.hpp"
#include "llama.h"

#include "ggml-backend.h"

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <map>
#include <mutex>
#include <random>
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
static model_catalog g_catalog;

static std::string make_id(const char * prefix) {
    static std::mt19937_64 rng{ std::random_device{}() };
    return std::string(prefix) + "-" + std::to_string(rng());
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
    fprintf(stderr, "usage: %s --model PATH [--listen HOST:PORT]\n", prog);
}

int main(int argc, char ** argv) {
    std::string listen = "0.0.0.0:9000";

    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "--listen") == 0 && i + 1 < argc) {
            listen = argv[++i];
        } else if (strcmp(argv[i], "--model") == 0 && i + 1 < argc) {
            g_model_path = argv[++i];
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

        if (body.contains("hardware")) {
            const auto & hw = body["hardware"];
            node.hardware.cpu_threads = hw.value("cpu_threads", 4);
            node.hardware.ram_gb      = hw.value("ram_gb", 0);
            node.hardware.gpu_name    = hw.value("gpu_name", "none");
            node.hardware.gpu_vram_gb = hw.value("gpu_vram_gb", 0);
        }

        if (body.contains("capabilities")) {
            const auto & caps = body["capabilities"];
            node.caps.gpu_backend   = caps.value("gpu_backend", "cpu");
            node.caps.gpu_memory_mb = caps.value("gpu_memory_mb", 0);
            node.caps.cpu_threads   = caps.value("cpu_threads", 4);
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

        fprintf(stderr, "orchestrator: registered node %s at %s:%d layers=%d score=%.1f decode=%.1f prefill=%.1f\n",
                node.node_id.c_str(), node.host.c_str(), node.http_port, node.n_layer, node.score,
                node.performance.decode_tps, node.performance.prefill_tps);

        res.set_content(json({ { "ok", true }, { "node_id", node.node_id } }).dump(), "application/json");
    });

    svr.Get("/nodes", [](const httplib::Request &, httplib::Response & res) {
        json nodes = json::array();
        std::lock_guard<std::mutex> lock(g_mu);
        for (const auto & kv : g_nodes) {
            const auto & n = kv.second;
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
                { "hardware", {
                    { "cpu_threads", n.hardware.cpu_threads },
                    { "ram_gb", n.hardware.ram_gb },
                    { "gpu_name", n.hardware.gpu_name },
                    { "gpu_vram_gb", n.hardware.gpu_vram_gb },
                }},
                { "performance", {
                    { "score", n.performance.score },
                    { "decode_tps", n.performance.decode_tps },
                    { "prefill_tps", n.performance.prefill_tps },
                    { "load_ms", n.performance.load_ms },
                }},
            });
        }
        res.set_content(json({ { "nodes", nodes } }).dump(), "application/json");
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

        std::vector<dist_planner_node> planner_nodes;
        planner_nodes.reserve(node_map.size());
        for (const auto & kv : node_map) {
            planner_nodes.push_back({ kv.second.node_id, kv.second.score });
        }

        const auto assignments = dist_plan_layers(n_layers, planner_nodes);
        if (assignments.empty()) {
            res.status = 500;
            res.set_content(R"({"error":"layer planning failed"})", "application/json");
            return;
        }

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
        session.model      = model;
        session.model_path = g_model_path;

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

        res.set_content(json({
            { "session_id", session.session_id },
            { "layout", layout_json(session) },
            { "pipeline", pipeline_json(session) },
        }).dump(), "application/json");
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
        json catalog_json = json::array();
        
        std::lock_guard<std::mutex> lock(g_mu);
        auto models = g_catalog.get_models();
        
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
    
    // GET /models - List installed models across cluster
    svr.Get("/models", [](const httplib::Request &, httplib::Response & res) {
        json models_json = json::array();
        
        std::lock_guard<std::mutex> lock(g_mu);
        
        // TODO: Query all nodes for their installed models
        // For now, return empty array
        
        res.set_content(models_json.dump(), "application/json");
    });
    
    // POST /models/install - Install a model on the cluster
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
        
        std::lock_guard<std::mutex> lock(g_mu);
        
        // Check if model exists in catalog
        const auto * model_info = g_catalog.find_model(model_id);
        if (!model_info) {
            res.status = 404;
            res.set_content(R"({"error":"model not found in catalog"})", "application/json");
            return;
        }
        
        // Create installation job
        std::string job_id = g_catalog.create_install_job(model_id);
        
        res.set_content(json({
            { "job_id", job_id },
            { "model", model_id },
            { "status", "started" }
        }).dump(), "application/json");
        
        // Get all online nodes (before async thread to avoid mutex deadlock)
        std::vector<dist_node_info> online_nodes;
        {
            std::lock_guard<std::mutex> nodes_lock(g_mu);
            for (const auto & [node_id, node] : g_nodes) {
                if (node.online) {
                    online_nodes.push_back(node);
                }
            }
        }
        
        // Trigger installation on all registered nodes (async)
        std::thread([job_id, model_id, model_info, online_nodes]() {
            
            if (online_nodes.empty()) {
                std::lock_guard<std::mutex> lock(g_mu);
                g_catalog.update_job_status(job_id, install_status::error, "No online nodes available");
                return;
            }
            
            {
                std::lock_guard<std::mutex> lock(g_mu);
                g_catalog.update_job_status(job_id, install_status::downloading, "", 0.1);
            }
            
            // Send installation request to each node
            int completed_nodes = 0;
            for (const auto & node : online_nodes) {
                httplib::Client client(node.host, node.http_port);
                client.set_connection_timeout(5, 0); // 5 seconds
                
                json install_request = {
                    { "model", model_id },
                    { "repo", model_info->source.repo },
                    { "file", model_info->source.file }
                };
                
                auto result = client.Post("/models/install", 
                                        install_request.dump(), 
                                        "application/json");
                
                if (result && result->status == 200) {
                    completed_nodes++;
                    printf("Installation triggered on %s:%d\n", node.host.c_str(), node.http_port);
                } else {
                    printf("Failed to trigger installation on %s:%d\n", node.host.c_str(), node.http_port);
                }
                
                // Update progress
                {
                    std::lock_guard<std::mutex> lock(g_mu);
                    double progress = 0.1 + (0.9 * completed_nodes / static_cast<double>(online_nodes.size()));
                    g_catalog.update_job_status(job_id, install_status::downloading, "", progress);
                }
            }
            
            // Final status update
            std::lock_guard<std::mutex> lock(g_mu);
            if (completed_nodes == static_cast<int>(online_nodes.size())) {
                g_catalog.complete_job(job_id);
                printf("Model installation completed on all %d nodes\n", completed_nodes);
            } else {
                g_catalog.update_job_status(job_id, install_status::error, 
                    "Installation failed on some nodes (" + 
                    std::to_string(completed_nodes) + "/" + 
                    std::to_string(online_nodes.size()) + " succeeded)");
            }
        }).detach();
    });
    
    // GET /models/install/{job_id} - Check installation status
    svr.Get(R"(/models/install/(.+))", [](const httplib::Request & req, httplib::Response & res) {
        const std::string job_id = req.matches[1];
        
        std::lock_guard<std::mutex> lock(g_mu);
        
        auto * job = g_catalog.get_install_job(job_id);
        if (!job) {
            res.status = 404;
            res.set_content(R"({"error":"job not found"})", "application/json");
            return;
        }
        
        std::string status_str;
        switch (job->status) {
            case install_status::unknown: status_str = "unknown"; break;
            case install_status::downloading: status_str = "downloading"; break;
            case install_status::ready: status_str = "ready"; break;
            case install_status::error: status_str = "error"; break;
        }
        
        json response = {
            { "job_id", job->job_id },
            { "model", job->model_id },
            { "status", status_str },
            { "progress", job->progress }
        };
        
        if (!job->error_msg.empty()) {
            response["error"] = job->error_msg;
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
