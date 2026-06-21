#include "dist_common.h"
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
    int layer_a_end     = DIST_LAYOUT_A_END;
    bool active         = false;
};

static std::mutex g_mu;
static std::map<std::string, dist_node_info> g_nodes;
static std::map<std::string, dist_session> g_sessions;
static std::string g_model_path;

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
        int timeout_ms = 30000) {
    httplib::Client cli(node.host.c_str(), node.http_port);
    cli.set_connection_timeout(5, 0);
    cli.set_read_timeout(timeout_ms / 1000, (timeout_ms % 1000) * 1000);

    const auto res = cli.Post("/configure", body.dump(), "application/json");
    if (!res || res->status != 200) {
        return false;
    }

    try {
        const json j = json::parse(res->body);
        return j.value("ok", false);
    } catch (...) {
        return false;
    }
}

static bool setup_pipeline(
        const std::vector<dist_node_info *> & picked,
        dist_session & session,
        std::string & err) {
    if (picked.size() != 3) {
        err = "need exactly 3 nodes";
        return false;
    }

    const int base = 9100 + (int) (getpid() % 500);
    const int pipe_base = base + 10;

    dist_pipeline_stage stage_c{};
    stage_c.node_id     = picked[2]->node_id;
    stage_c.host        = picked[2]->host;
    stage_c.http_port   = picked[2]->http_port;
    stage_c.peer_port   = pipe_base + 3;
    stage_c.layer_start = DIST_LAYOUT_C_START;
    stage_c.layer_end   = picked[2]->n_layer;
    stage_c.role        = DIST_ROLE_FINAL;

    dist_pipeline_stage stage_b{};
    stage_b.node_id     = picked[1]->node_id;
    stage_b.host        = picked[1]->host;
    stage_b.http_port   = picked[1]->http_port;
    stage_b.peer_port   = pipe_base + 2;
    stage_b.layer_start = DIST_LAYOUT_B_START;
    stage_b.layer_end   = DIST_LAYOUT_B_END;
    stage_b.role        = DIST_ROLE_MIDDLE;

    dist_pipeline_stage stage_a{};
    stage_a.node_id     = picked[0]->node_id;
    stage_a.host        = picked[0]->host;
    stage_a.http_port   = picked[0]->http_port;
    stage_a.ctrl_port   = pipe_base + 1;
    stage_a.layer_start = 0;
    stage_a.layer_end   = DIST_LAYOUT_A_END;
    stage_a.role        = DIST_ROLE_ENTRY;

    json cfg_c = {
        { "role", "final" },
        { "layer_start", stage_c.layer_start },
        { "layer_end", stage_c.layer_end },
        { "peer_port", stage_c.peer_port },
        { "peer_bind", "0.0.0.0" },
    };

    if (!configure_node(*picked[2], cfg_c)) {
        err = "failed to configure final node " + stage_c.node_id;
        return false;
    }

#if !defined(_WIN32)
    usleep(200000);
#endif

    json cfg_b = {
        { "role", "middle" },
        { "layer_start", stage_b.layer_start },
        { "layer_end", stage_b.layer_end },
        { "peer_port", stage_b.peer_port },
        { "peer_bind", "0.0.0.0" },
        { "next_host", stage_c.host },
        { "next_port", stage_c.peer_port },
    };

    if (!configure_node(*picked[1], cfg_b)) {
        err = "failed to configure middle node " + stage_b.node_id;
        return false;
    }

#if !defined(_WIN32)
    usleep(200000);
#endif

    json cfg_a = {
        { "role", "entry" },
        { "layer_start", stage_a.layer_start },
        { "layer_end", stage_a.layer_end },
        { "ctrl_port", stage_a.ctrl_port },
        { "next_host", stage_b.host },
        { "next_port", stage_b.peer_port },
    };

    if (!configure_node(*picked[0], cfg_a)) {
        err = "failed to configure entry node " + stage_a.node_id;
        return false;
    }

#if !defined(_WIN32)
    usleep(500000);
#endif

    session.pipeline = { stage_a, stage_b, stage_c };
    session.entry_host       = stage_a.host;
    session.entry_ctrl_port  = stage_a.ctrl_port;
    session.layer_a_end      = stage_a.layer_end;
    session.active           = true;
    return true;
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

    if (!gen3_send_recv(ctrl_fd, SPLIT_GEN_CMD_RESET, 0, 0, session.layer_a_end, nullptr, resp)) {
        err = "reset failed";
        close(ctrl_fd);
        return false;
    }

    std::vector<int32_t> ptoks(prompt.size());
    for (size_t i = 0; i < prompt.size(); ++i) {
        ptoks[i] = (int32_t) prompt[i];
    }

    if (!gen3_send_recv(ctrl_fd, SPLIT_GEN_CMD_PREFILL, (int32_t) prompt.size(), 0,
            session.layer_a_end, ptoks.data(), resp)) {
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

        if (!gen3_send_recv(ctrl_fd, SPLIT_GEN_CMD_DECODE, 1, pos, session.layer_a_end, &tok_i32, resp)) {
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

    gen3_send_recv(ctrl_fd, SPLIT_GEN_CMD_SHUTDOWN, 0, 0, session.layer_a_end, nullptr, resp);
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
        node.online = true;

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

        fprintf(stderr, "orchestrator: registered node %s at %s:%d layers=%d\n",
                node.node_id.c_str(), node.host.c_str(), node.http_port, node.n_layer);

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
                { "online", n.online },
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

        std::vector<dist_node_info *> picked;
        {
            std::lock_guard<std::mutex> lock(g_mu);
            for (auto & kv : g_nodes) {
                if (kv.second.online && kv.second.n_layer > 0) {
                    picked.push_back(&kv.second);
                }
            }
        }

        std::sort(picked.begin(), picked.end(), [](const dist_node_info * a, const dist_node_info * b) {
            return a->node_id < b->node_id;
        });

        if (picked.size() < 3) {
            res.status = 503;
            res.set_content(json({
                { "error", "need at least 3 registered nodes" },
                { "registered", picked.size() },
            }).dump(), "application/json");
            return;
        }

        picked.resize(3);

        dist_session session{};
        session.session_id = make_id("sess");
        session.model      = model;
        session.model_path = g_model_path;

        std::string err;
        if (!setup_pipeline(picked, session, err)) {
            res.status = 500;
            res.set_content(json({ { "error", err } }).dump(), "application/json");
            return;
        }

        {
            std::lock_guard<std::mutex> lock(g_mu);
            g_sessions[session.session_id] = session;
        }

        res.set_content(json({
            { "session_id", session.session_id },
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

    fprintf(stderr, "orchestrator: listening on %s:%d\n", bind_host.c_str(), port);

    if (!svr.listen(bind_host.c_str(), port)) {
        fprintf(stderr, "orchestrator: failed to bind\n");
        return 1;
    }

    return 0;
}
