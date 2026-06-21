#include "node_core.h"

#include "ggml-backend.h"
#include "ggml.h"

#include <cstdio>
#include <cstring>
#include <vector>

#if !defined(_WIN32)
#include <unistd.h>
#endif

static void dist_close_fd(int & fd) {
    if (fd >= 0) {
#if !defined(_WIN32)
        close(fd);
#endif
        fd = -1;
    }
}

bool dist_node_runtime::load_model() {
    ggml_backend_load_all();

    model = llama_model_load_from_file(model_path.c_str(), llama_model_default_params());
    if (!model) {
        return false;
    }

    n_embd  = llama_model_n_embd(model);
    n_layer = llama_model_n_layer(model);
    n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model));

    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx   = 512;
    cparams.n_batch = 512;
    cparams.no_perf = true;

    ctx = llama_init_from_model(model, cparams);
    if (!ctx) {
        return false;
    }

    smpl = split_gen_make_sampler();
    return true;
}

void dist_node_runtime::unload_model() {
    shutdown_pipeline();
    if (smpl) {
        llama_sampler_free(smpl);
        smpl = nullptr;
    }
    if (ctx) {
        llama_free(ctx);
        ctx = nullptr;
    }
    if (model) {
        llama_model_free(model);
        model = nullptr;
    }
}

void dist_node_runtime::shutdown_pipeline() {
    stop = true;
    if (peer_thread.joinable()) {
        peer_thread.join();
    }

    if (next_fd >= 0 && cfg.role == DIST_ROLE_ENTRY) {
        split_ab_send_shutdown(next_fd);
    }
    if (next_fd >= 0 && cfg.role == DIST_ROLE_MIDDLE) {
        split_ab_send_shutdown(next_fd);
    }

    dist_close_fd(ctrl_fd);
    dist_close_fd(next_fd);
    dist_close_fd(peer_fd);
    dist_close_fd(listen_fd);

    configured = false;
    peer_ready = false;
    stop = false;
}

bool dist_node_runtime::configure(const dist_configure_req & req, std::string & err) {
    shutdown_pipeline();
    cfg = req;

    if (cfg.role == DIST_ROLE_ENTRY) {
        if (cfg.ctrl_port <= 0 || cfg.next_port <= 0 || cfg.next_host.empty()) {
            err = "entry node requires ctrl_port, next_host, next_port";
            return false;
        }

        next_fd = split_tcp_connect_retry(cfg.next_host.c_str(), cfg.next_port, 300, 100);
        if (next_fd < 0) {
            err = "failed to connect to next node";
            return false;
        }

        listen_fd = split_tcp_listen_host("0.0.0.0", cfg.ctrl_port);
        if (listen_fd < 0) {
            err = "failed to listen on ctrl_port";
            dist_close_fd(next_fd);
            return false;
        }

        configured = true;
        stop = false;
        peer_thread = std::thread([this]() { entry_ctrl_loop(); });
        peer_ready = true;
        return true;
    }

    if (cfg.role == DIST_ROLE_MIDDLE || cfg.role == DIST_ROLE_FINAL) {
        if (cfg.peer_port <= 0) {
            err = "middle/final node requires peer_port";
            return false;
        }

        if (cfg.role == DIST_ROLE_MIDDLE) {
            if (cfg.next_port <= 0 || cfg.next_host.empty()) {
                err = "middle node requires next_host and next_port";
                return false;
            }
            next_fd = split_tcp_connect_retry(cfg.next_host.c_str(), cfg.next_port, 300, 100);
            if (next_fd < 0) {
                err = "failed to connect to next node";
                return false;
            }
        }

        listen_fd = split_tcp_listen_host(cfg.peer_bind.c_str(), cfg.peer_port);
        if (listen_fd < 0) {
            err = "failed to listen on peer_port";
            dist_close_fd(next_fd);
            return false;
        }

        configured = true;
        stop = false;
        peer_thread = std::thread([this]() { peer_loop(); });
        return true;
    }

    err = "invalid role";
    return false;
}

static bool dist_run_b_layers(
        llama_context * ctx,
        const split_tcp_hidden_msg & msg,
        int32_t n_embd,
        int layer_start,
        int layer_end,
        double & ms_b_out,
        std::vector<float> & out_hidden) {
    llama_set_layer_range(ctx, layer_start, layer_end);

    const int32_t n_tokens  = msg.header.n_tokens;
    const int32_t pos_start = msg.meta.pos_start;

    const int64_t t0 = ggml_time_us();

    if (n_tokens <= 1) {
        llama_set_hidden_state(ctx, msg.data.data(), n_tokens);
        if (split_gen_decode_hidden(ctx, msg.data.data(), n_tokens, n_embd, pos_start, true) != 0) {
            return false;
        }
        const float * h = llama_get_embeddings(ctx);
        if (h == nullptr) {
            return false;
        }
        out_hidden.assign(h, h + (size_t) n_tokens * n_embd);
    } else {
        out_hidden.resize((size_t) n_tokens * n_embd);
        for (int32_t i = 0; i < n_tokens; ++i) {
            const float * in = msg.data.data() + (size_t) i * n_embd;
            llama_set_hidden_state(ctx, in, 1);
            if (split_gen_decode_hidden(ctx, in, 1, n_embd, pos_start + i, true) != 0) {
                return false;
            }
            const float * h = llama_get_embeddings(ctx);
            if (h == nullptr) {
                return false;
            }
            std::memcpy(out_hidden.data() + (size_t) i * n_embd, h, (size_t) n_embd * sizeof(float));
        }
    }

    ms_b_out = (ggml_time_us() - t0) / 1000.0;
    return true;
}

bool dist_node_forward_to_next(
        dist_node_runtime & rt,
        int32_t n_tokens,
        int32_t pos_start,
        bool include_logits,
        double ms_local,
        dist_gen_resp & out) {
    if (rt.next_fd < 0) {
        out.error = "next node not connected";
        return false;
    }

    const float * hidden = llama_get_embeddings(rt.ctx);
    if (hidden == nullptr && n_tokens > 0) {
        out.error = "missing hidden state";
        return false;
    }

    const int64_t t0 = ggml_time_us();
    if (!split_ab_send_hidden(rt.next_fd, n_tokens, rt.n_embd, rt.cfg.layer_end, pos_start,
            include_logits ? 1 : 0, hidden)) {
        out.error = "failed to send hidden to next node";
        return false;
    }
    const int64_t t1 = ggml_time_us();

    if (rt.cfg.role == DIST_ROLE_ENTRY) {
        split_gen3_mid_resp mid{};
        if (!split_gen3_recv_mid_resp(rt.next_fd, mid)) {
            out.error = "failed to receive response from pipeline";
            return false;
        }
        out.ok             = true;
        out.token_id       = mid.token_id;
        out.ms_a_compute   = ms_local;
        out.ms_ab_xfer     = (t1 - t0) / 1000.0;
        out.ms_b_compute   = mid.ms_b_compute;
        out.ms_bc_xfer     = mid.ms_bc_xfer;
        out.ms_c_compute   = mid.ms_c_compute;
        out.ms_c_sample    = mid.ms_c_sample;
        return true;
    }

    split_gen3_c_resp cresp{};
    if (!split_gen3_recv_c_resp(rt.next_fd, cresp)) {
        out.error = "failed to receive response from final node";
        return false;
    }

    out.ok             = true;
    out.token_id       = cresp.token_id;
    out.ms_b_compute   = ms_local;
    out.ms_bc_xfer     = (t1 - t0) / 1000.0;
    out.ms_c_compute   = cresp.ms_compute;
    out.ms_c_sample    = cresp.ms_sample;
    return true;
}

dist_gen_resp dist_node_runtime::handle_entry_cmd(
        split_gen_cmd cmd,
        int32_t n_tokens,
        int32_t pos_start,
        const int32_t * tokens) {
    dist_gen_resp out{};

    if (!configured.load() || cfg.role != DIST_ROLE_ENTRY) {
        out.error = "node not configured as entry";
        return out;
    }

    std::lock_guard<std::mutex> lock(llama_mu);
    llama_set_layer_range(ctx, cfg.layer_start, cfg.layer_end);

    if (cmd == SPLIT_GEN_CMD_RESET) {
        llama_memory_clear(llama_get_memory(ctx), true);
        split_ab_send_reset(next_fd);
        out.ok = true;
        out.token_id = -1;
        return out;
    }

    if (cmd == SPLIT_GEN_CMD_SHUTDOWN) {
        split_ab_send_shutdown(next_fd);
        out.ok = true;
        return out;
    }

    const int64_t t0 = ggml_time_us();
    int decode_rc = 0;

    if (cmd == SPLIT_GEN_CMD_PREFILL) {
        std::vector<llama_token> toks((size_t) n_tokens);
        for (int32_t i = 0; i < n_tokens; ++i) {
            toks[i] = (llama_token) tokens[i];
        }
        decode_rc = split_gen_decode_tokens(ctx, toks, pos_start, true);
    } else if (cmd == SPLIT_GEN_CMD_DECODE) {
        decode_rc = split_gen_decode_one(ctx, (llama_token) tokens[0], pos_start);
    } else {
        out.error = "unknown command";
        return out;
    }

    const double ms_local = (ggml_time_us() - t0) / 1000.0;

    if (decode_rc != 0) {
        out.error = "decode failed";
        return out;
    }

    const int32_t n_out = (cmd == SPLIT_GEN_CMD_PREFILL) ? n_tokens : 1;
    if (!dist_node_forward_to_next(*this, n_out, pos_start, false, ms_local, out)) {
        return out;
    }

    return out;
}

void dist_node_runtime::entry_ctrl_loop() {
    ctrl_fd = split_tcp_accept(listen_fd);
#if !defined(_WIN32)
    close(listen_fd);
#endif
    listen_fd = -1;

    if (ctrl_fd < 0) {
        peer_ready = false;
        return;
    }

    split_tcp_set_timeouts(ctrl_fd, 30000);

    peer_ready = true;

    while (!stop.load()) {
        split_gen_a_req req{};
        std::vector<int32_t> tokens_i32;
        if (!split_gen_recv_req(ctrl_fd, req, tokens_i32)) {
            break;
        }

        if (req.cmd == SPLIT_GEN_CMD_SHUTDOWN) {
            handle_entry_cmd(SPLIT_GEN_CMD_SHUTDOWN, 0, 0, nullptr);
            split_gen3_a_resp resp{};
            resp.magic   = SPLIT_GEN_MAGIC;
            resp.version = SPLIT_GEN3_VERSION;
            split_gen3_send_a_resp(ctrl_fd, resp, nullptr, 0);
            break;
        }

        dist_gen_resp gen = handle_entry_cmd(
                (split_gen_cmd) req.cmd,
                req.n_tokens,
                req.pos_start,
                tokens_i32.empty() ? nullptr : tokens_i32.data());

        split_gen3_a_resp resp{};
        resp.magic          = SPLIT_GEN_MAGIC;
        resp.version        = SPLIT_GEN3_VERSION;
        resp.token_id       = gen.token_id;
        resp.ms_a_compute   = gen.ms_a_compute;
        resp.ms_ab_xfer     = gen.ms_ab_xfer;
        resp.ms_b_compute   = gen.ms_b_compute;
        resp.ms_bc_xfer     = gen.ms_bc_xfer;
        resp.ms_c_compute   = gen.ms_c_compute;
        resp.ms_c_sample    = gen.ms_c_sample;

        if (!gen.ok && !gen.error.empty()) {
            resp.token_id = -2;
        }

        if (!split_gen3_send_a_resp(ctrl_fd, resp, nullptr, 0)) {
            break;
        }

        if (!gen.ok && req.cmd != SPLIT_GEN_CMD_RESET) {
            break;
        }
    }

#if !defined(_WIN32)
    close(ctrl_fd);
#endif
    ctrl_fd = -1;
}

bool dist_node_run_middle_step(dist_node_runtime & rt, int peer_fd) {
    split_ab_cmd cmd;
    if (!split_ab_recv_cmd(peer_fd, cmd)) {
        return false;
    }

    if (cmd == SPLIT_AB_CMD_SHUTDOWN) {
        split_ab_send_shutdown(rt.next_fd);
        return false;
    }

    if (cmd == SPLIT_AB_CMD_RESET) {
        std::lock_guard<std::mutex> lock(rt.llama_mu);
        llama_memory_clear(llama_get_memory(rt.ctx), true);
        llama_clear_hidden_state(rt.ctx);
        split_ab_send_reset(rt.next_fd);
        return true;
    }

    if (cmd != SPLIT_AB_CMD_HIDDEN) {
        return false;
    }

    split_tcp_hidden_msg msg;
    if (!split_ab_recv_hidden(peer_fd, msg)) {
        return false;
    }

    double ms_b = 0.0;
    std::vector<float> out_hidden;
    {
        std::lock_guard<std::mutex> lock(rt.llama_mu);
        llama_set_layer_range(rt.ctx, rt.cfg.layer_start, rt.cfg.layer_end);
        if (!dist_run_b_layers(rt.ctx, msg, rt.n_embd, rt.cfg.layer_start, rt.cfg.layer_end, ms_b, out_hidden)) {
            return false;
        }
    }

    dist_gen_resp fwd{};
    if (!dist_node_forward_to_next(rt, msg.header.n_tokens, msg.meta.pos_start, false, ms_b, fwd)) {
        return false;
    }

    split_gen3_mid_resp resp{};
    resp.magic        = SPLIT_GEN_MAGIC;
    resp.token_id     = fwd.token_id;
    resp.n_vocab      = rt.n_vocab;
    resp.ms_b_compute = fwd.ms_b_compute;
    resp.ms_bc_xfer   = fwd.ms_bc_xfer;
    resp.ms_c_compute = fwd.ms_c_compute;
    resp.ms_c_sample  = fwd.ms_c_sample;

    return split_gen3_send_mid_resp(peer_fd, resp);
}

bool dist_node_run_final_step(dist_node_runtime & rt, int peer_fd) {
    split_ab_cmd cmd;
    if (!split_ab_recv_cmd(peer_fd, cmd)) {
        return false;
    }

    if (cmd == SPLIT_AB_CMD_SHUTDOWN) {
        return false;
    }

    if (cmd == SPLIT_AB_CMD_RESET) {
        std::lock_guard<std::mutex> lock(rt.llama_mu);
        llama_memory_clear(llama_get_memory(rt.ctx), true);
        llama_clear_hidden_state(rt.ctx);
        return true;
    }

    if (cmd != SPLIT_AB_CMD_HIDDEN) {
        return false;
    }

    split_tcp_hidden_msg msg;
    if (!split_ab_recv_hidden(peer_fd, msg)) {
        return false;
    }

    std::lock_guard<std::mutex> lock(rt.llama_mu);
    llama_set_layer_range(rt.ctx, rt.cfg.layer_start, rt.n_layer);
    llama_set_hidden_state(rt.ctx, msg.data.data(), msg.header.n_tokens);

    const int64_t t0 = ggml_time_us();
    if (split_gen_decode_hidden(rt.ctx, msg.data.data(), msg.header.n_tokens, msg.header.n_embd,
            msg.meta.pos_start, true) != 0) {
        return false;
    }
    const double ms_compute = (ggml_time_us() - t0) / 1000.0;

    const int64_t t1 = ggml_time_us();
    const llama_token token_id = llama_sampler_sample(rt.smpl, rt.ctx, -1);
    llama_sampler_accept(rt.smpl, token_id);
    const double ms_sample = (ggml_time_us() - t1) / 1000.0;

    split_gen3_c_resp resp{};
    resp.magic      = SPLIT_GEN_MAGIC;
    resp.token_id   = (int32_t) token_id;
    resp.n_vocab    = rt.n_vocab;
    resp.ms_compute = ms_compute;
    resp.ms_sample  = ms_sample;

    return split_gen3_send_c_resp(peer_fd, resp);
}

void dist_node_runtime::peer_loop() {
    peer_fd = split_tcp_accept(listen_fd);
#if !defined(_WIN32)
    close(listen_fd);
#endif
    listen_fd = -1;

    if (peer_fd < 0) {
        peer_ready = false;
        return;
    }

    peer_ready = true;

    while (!stop.load()) {
        bool ok = false;
        if (cfg.role == DIST_ROLE_MIDDLE) {
            ok = dist_node_run_middle_step(*this, peer_fd);
        } else if (cfg.role == DIST_ROLE_FINAL) {
            ok = dist_node_run_final_step(*this, peer_fd);
        }

        if (!ok) {
            break;
        }
    }

#if !defined(_WIN32)
    close(peer_fd);
#endif
    peer_fd = -1;
    peer_ready = false;
}
