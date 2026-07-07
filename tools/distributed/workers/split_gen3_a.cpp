// Process A: entry stage - layers [0, layer_end)

#include "ggml-backend.h"
#include "ggml.h"
#include "split_gen_model_load.h"
#include "split_gen_common.h"
#include "../transport/split_tcp_wire.h"
#include "../runtime_debug/debug_hooks.h"
#include "../runtime_debug/runtime_debug.h"
#include "../runtime_debug/trace_recorder.h"
#include "../runtime_debug/hidden_transport.h"
#include "../runtime_debug/perf_trace.h"
#include "../runtime_debug/perf_ggml.h"

#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

static void usage(const char * prog) {
    fprintf(stderr,
            "usage: %s MODEL --ctrl-port PORT --b-port PORT [--layer-end N] "
            "[--ready-file PATH]\n",
            prog);
}

static bool gather_stage_hidden(
        llama_context * ctx,
        int32_t n_tokens,
        int32_t n_embd,
        std::vector<float> & out) {
    if (n_tokens <= 0) {
        out.clear();
        return true;
    }
    out.resize((size_t) n_tokens * (size_t) n_embd);
    for (int32_t i = 0; i < n_tokens; ++i) {
        const float * h = n_tokens > 1 ? llama_get_embeddings_ith(ctx, i) : llama_get_embeddings(ctx);
        if (h == nullptr) {
            return false;
        }
        std::memcpy(
                out.data() + (size_t) i * (size_t) n_embd,
                h,
                (size_t) n_embd * sizeof(float));
    }
    return true;
}

static bool forward_to_peer(
        int peer_fd,
        llama_context * ctx,
        int32_t n_tokens,
        int32_t n_embd,
        int32_t layer_end,
        int32_t pos_start,
        bool include_logits,
        int32_t n_vocab,
        double ms_a,
        bool next_is_final,
        int32_t debug_step,
        const char * phase,
        trace_recorder * dbg,
        split_gen3_a_resp & resp,
        std::vector<float> & logits_out) {
    perf_trace_refresh_context();
    const int32_t tok_idx = (phase && std::strcmp(phase, "decode") == 0) ? debug_step : -1;

    std::vector<float> hidden_buf;
    const int64_t t_ser0 = ggml_time_us();
    if (!gather_stage_hidden(ctx, n_tokens, n_embd, hidden_buf)) {
        fprintf(stderr, "gen3_a: missing hidden state\n");
        return false;
    }
    const float * hidden = hidden_buf.data();
    const int64_t serialize_us = ggml_time_us() - t_ser0;
    const int32_t payload_bytes = n_tokens * n_embd * (int32_t) sizeof(float);

    hidden_transport_trace tr_send = dist_debug_transport_send(
            debug_step, phase, "ab", n_tokens, n_embd, layer_end, pos_start, hidden, 0.0);
    if (dbg) {
        dbg->emit_transport(tr_send);
    }

    const int64_t t0 = ggml_time_us();
    split_gen_pipe_trace("entry", phase, "send_hidden_enter", n_tokens, n_embd, pos_start, -1, layer_end);
    if (!split_ab_send_hidden(peer_fd, n_tokens, n_embd, layer_end, pos_start,
            include_logits ? 1 : 0, hidden)) {
        fprintf(stderr, "gen3_a: send hidden failed\n");
        return false;
    }
    const int64_t send_us = ggml_time_us() - t0;
    split_gen_pipe_trace("entry", phase, "send_hidden_exit", n_tokens, n_embd, pos_start, -1, layer_end);

    if (perf_trace_enabled()) {
        perf_trace_set_component("entry");
        perf_emit_hidden_transfer("entry", tok_idx, "ab", payload_bytes, serialize_us, send_us, 0, 0);
        perf_emit_span("ENTRY_SEND_END", perf_category::NETWORK, "entry", tok_idx, send_us, nullptr);
    }

    resp.magic          = SPLIT_GEN_MAGIC;
    resp.version        = SPLIT_GEN3_VERSION;
    resp.include_logits = include_logits ? 1 : 0;
    resp.n_vocab        = n_vocab;
    resp.ms_a_compute   = ms_a;
    resp.ms_ab_xfer     = send_us / 1000.0;
    logits_out.clear();

    if (next_is_final) {
        split_gen3_c_resp cresp{};
        if (!split_gen3_recv_c_resp(peer_fd, cresp)) {
            fprintf(stderr, "gen3_a: recv final resp failed\n");
            return false;
        }
        resp.token_id     = cresp.token_id;
        resp.ms_b_compute = 0.0;
        resp.ms_bc_xfer   = 0.0;
        resp.ms_c_compute = cresp.ms_compute;
        resp.ms_c_sample  = cresp.ms_sample;
        return true;
    }

    split_gen3_mid_resp mid{};
    split_gen_pipe_trace("entry", phase, "recv_mid_enter", n_tokens, n_embd, pos_start, -1, layer_end);
    if (!split_gen3_recv_mid_resp(peer_fd, mid)) {
        fprintf(stderr, "gen3_a: recv mid resp failed\n");
        return false;
    }
    split_gen_pipe_trace("entry", phase, "recv_mid_exit", n_tokens, n_embd, pos_start, -1, layer_end);

    resp.token_id     = mid.token_id;
    resp.ms_b_compute = mid.ms_b_compute;
    resp.ms_bc_xfer   = mid.ms_bc_xfer;
    resp.ms_c_compute = mid.ms_c_compute;
    resp.ms_c_sample  = mid.ms_c_sample;
    return true;
}

int main(int argc, char ** argv) {
    split_tcp_init();
    if (argc < 2) {
        usage(argv[0]);
        return 1;
    }

    const char * model_path = argv[1];
    int ctrl_port = -1;
    int b_port    = -1;
    int layer_start = 0;
    int layer_end = 5;
    bool next_is_final = false;
    const char * b_host = "127.0.0.1";
    const char * bind_host = "0.0.0.0";
    std::string ready_file;

    for (int i = 2; i < argc; ++i) {
        if (strcmp(argv[i], "--ctrl-port") == 0 && i + 1 < argc) {
            ctrl_port = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--b-port") == 0 && i + 1 < argc) {
            b_port = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--b-host") == 0 && i + 1 < argc) {
            b_host = argv[++i];
        } else if (strcmp(argv[i], "--bind") == 0 && i + 1 < argc) {
            bind_host = argv[++i];
        } else if (strcmp(argv[i], "--layer-start") == 0 && i + 1 < argc) {
            layer_start = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--layer-end") == 0 && i + 1 < argc) {
            layer_end = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--next-final") == 0) {
            next_is_final = true;
        } else if (strcmp(argv[i], "--ready-file") == 0 && i + 1 < argc) {
            ready_file = argv[++i];
        } else {
            usage(argv[0]);
            return 1;
        }
    }

    if (ctrl_port < 0 || b_port < 0) {
        usage(argv[0]);
        return 1;
    }

    split_gen_write_ready_state(ready_file, "MODEL_LOADING");

    const int listen_fd = split_tcp_listen_host(bind_host, ctrl_port);
    if (listen_fd < 0) {
        split_gen_write_ready_state(ready_file, "FAILED");
        fprintf(stderr, "gen3_a: ctrl listen failed port=%d\n", ctrl_port);
        return 1;
    }
    split_gen_write_ready_state(ready_file, "LISTENER_READY");

    ggml_backend_load_all();

    std::string load_err;
    llama_model * model = split_gen_load_model(model_path, load_err);
    if (!model) {
        split_tcp_close(listen_fd);
        split_gen_write_ready_state(ready_file, "FAILED");
        fprintf(stderr, "gen3_a: load model failed: %s\n", load_err.c_str());
        return 1;
    }

    const int32_t n_embd  = llama_model_n_embd(model);
    const int32_t n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model));

    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx   = 512;
    cparams.n_batch = 512;
    cparams.n_ubatch = 1;
    cparams.no_perf = true;
    cparams.layer_start = layer_start;
    cparams.layer_end   = layer_end;
    fprintf(stderr, "gen3_a: create context layer_start=%d layer_end=%d\n", layer_start, layer_end);
    llama_context * ctx = llama_init_from_model(model, cparams);
    if (!ctx) {
        split_tcp_close(listen_fd);
        split_gen_write_ready_state(ready_file, "FAILED");
        fprintf(stderr, "gen3_a: create context failed\n");
        return 1;
    }

    llama_set_layer_range(ctx, layer_start, layer_end);

    dist_debug_load_config();
    dist_debug_reset_recorder("entry");
    trace_recorder * dbg = dist_debug_recorder();
    int32_t debug_step   = 0;

    const int b_fd = split_tcp_connect_retry(b_host, b_port, 3000, 100);
    if (b_fd < 0) {
        split_gen_write_ready_state(ready_file, "FAILED");
        fprintf(stderr, "gen3_a: connect to B failed %s:%d\n", b_host, b_port);
        split_tcp_close(listen_fd);
        return 1;
    }
    split_gen_write_ready_state(ready_file, "PIPE_READY");

    fprintf(stderr, "gen3_a: ready ctrl_port=%d layer_start=%d layer_end=%d\n",
            ctrl_port, layer_start, layer_end);
    split_gen_write_ready_state(ready_file, "READY");

    bool normal_shutdown = false;
    bool pipe_failed = false;
    while (true) {
        const int ctrl_fd = split_tcp_accept(listen_fd);
        if (ctrl_fd < 0) {
            fprintf(stderr, "gen3_a: ctrl accept failed\n");
            continue;
        }

        while (true) {
        split_gen_a_req req{};
        std::vector<int32_t> tokens_i32;
        std::vector<float> hidden_in;
        int32_t hidden_n_embd = 0;
        perf_trace_sched_queue_wait_begin();
        const bool got_req = split_gen_recv_req(ctrl_fd, req, tokens_i32, &hidden_in, &hidden_n_embd);
        perf_trace_sched_queue_wait_end();
        if (!got_req) {
            fprintf(stderr, "gen3_a: client disconnected, waiting for next ctrl connection\n");
            split_tcp_close(ctrl_fd);
            break;
        }

        if (req.cmd == SPLIT_GEN_CMD_SHUTDOWN) {
            split_ab_send_shutdown(b_fd);
            split_gen3_a_resp resp{};
            resp.magic   = SPLIT_GEN_MAGIC;
            resp.version = SPLIT_GEN3_VERSION;
            split_gen3_send_a_resp(ctrl_fd, resp, nullptr, 0);
            split_tcp_close(ctrl_fd);
            split_tcp_close(b_fd);
            split_tcp_close(listen_fd);

            llama_free(ctx);
            llama_model_free(model);
            split_gen_write_ready_state(ready_file, "STOPPED");
            return 0;
        }

        if (req.cmd == SPLIT_GEN_CMD_RESET) {
            llama_memory_clear(llama_get_memory(ctx), true);
            llama_clear_hidden_state(ctx);
            debug_step = 0;
            if (dbg) {
                dbg->emit_step_begin(0, "reset", -1, 0, 0);
            }
            split_ab_send_reset(b_fd);
            split_gen3_a_resp resp{};
            resp.magic   = SPLIT_GEN_MAGIC;
            resp.version = SPLIT_GEN3_VERSION;
            resp.token_id = -1;
            split_gen3_send_a_resp(ctrl_fd, resp, nullptr, 0);
            continue;
        }

        const int64_t t0 = ggml_time_us();
        int decode_rc = 0;
        const char * phase = (req.cmd == SPLIT_GEN_CMD_PREFILL ||
                              req.cmd == SPLIT_GEN_CMD_PREFILL_HIDDEN) ? "prefill" : "decode";
        const int32_t in_tok = tokens_i32.empty() ? -1 : tokens_i32[0];

        if (dbg) {
            dbg->emit_step_begin(debug_step, phase, in_tok, req.pos_start, 0);
            dist_debug_log_position(dbg, debug_step, phase, ctx, req.pos_start,
                    req.cmd == SPLIT_GEN_CMD_PREFILL || req.cmd == SPLIT_GEN_CMD_PREFILL_HIDDEN
                            ? req.n_tokens : 1,
                    1);
        }

        split_gen_pipe_trace("entry", phase, "decode_enter", req.n_tokens, hidden_n_embd > 0 ? hidden_n_embd : n_embd, req.pos_start, layer_start, layer_end);
        const int32_t tok_idx = (phase && std::strcmp(phase, "decode") == 0) ? debug_step : -1;
        const bool decode_step = (phase && std::strcmp(phase, "decode") == 0);
        static int32_t entry_queue_depth = 0;
        if (decode_step && perf_trace_enabled()) {
            perf_trace_refresh_context();
            perf_trace_set_component("entry");
            perf_emit_instant("ENTRY_RECEIVE", perf_category::NETWORK, "entry", tok_idx, nullptr);
            perf_emit_queue_depth("entry", tok_idx, entry_queue_depth);
        }
        if (decode_step) {
            entry_queue_depth = std::max(0, entry_queue_depth - 1);
        }
        perf_span compute_span("ENTRY_COMPUTE_BEGIN", "ENTRY_COMPUTE_END", perf_category::COMPUTE, "entry");
        compute_span.set_token_idx(tok_idx);
        if (req.cmd == SPLIT_GEN_CMD_PREFILL) {
            if (layer_start > 0) {
                fprintf(stderr, "gen3_a: token prefill requires layer_start=0\n");
                split_tcp_close(ctrl_fd);
                break;
            }
            std::vector<llama_token> tokens((size_t) req.n_tokens);
            for (int32_t i = 0; i < req.n_tokens; ++i) {
                tokens[i] = (llama_token) tokens_i32[i];
            }
            decode_rc = split_gen_decode_tokens(ctx, tokens, req.pos_start, true);
        } else if (req.cmd == SPLIT_GEN_CMD_DECODE) {
            if (layer_start > 0) {
                fprintf(stderr, "gen3_a: token decode requires layer_start=0\n");
                split_tcp_close(ctrl_fd);
                break;
            }
            decode_rc = split_gen_decode_one(ctx, (llama_token) tokens_i32[0], req.pos_start);
        } else if (req.cmd == SPLIT_GEN_CMD_PREFILL_HIDDEN) {
            const int32_t n_emb = hidden_n_embd > 0 ? hidden_n_embd : n_embd;
            decode_rc = split_gen_decode_hidden(
                    ctx, hidden_in.data(), req.n_tokens, n_emb, req.pos_start, true,
                    layer_start > 0);
        } else if (req.cmd == SPLIT_GEN_CMD_DECODE_HIDDEN) {
            const int32_t n_emb = hidden_n_embd > 0 ? hidden_n_embd : n_embd;
            decode_rc = split_gen_decode_hidden(
                    ctx, hidden_in.data(), 1, n_emb, req.pos_start, true, layer_start > 0);
        } else {
            fprintf(stderr, "gen3_a: unknown cmd %u\n", req.cmd);
            split_tcp_close(ctrl_fd);
            break;
        }
        split_gen_pipe_trace("entry", phase, "decode_exit", req.n_tokens, hidden_n_embd > 0 ? hidden_n_embd : n_embd, req.pos_start, layer_start, layer_end);

        const double ms_a = (ggml_time_us() - t0) / 1000.0;

        if (decode_rc != 0) {
            fprintf(stderr, "gen3_a: decode failed cmd=%u\n", req.cmd);
            split_tcp_close(ctrl_fd);
            break;
        }

        const int32_t n_out = (req.cmd == SPLIT_GEN_CMD_PREFILL ||
                               req.cmd == SPLIT_GEN_CMD_PREFILL_HIDDEN) ? req.n_tokens : 1;
        const int32_t le    = req.layer_end > 0 ? req.layer_end : layer_end;

        if (dbg) {
            dist_debug_log_kv(dbg, debug_step, phase, ctx, 0);
            dist_debug_log_hidden_out(dbg, debug_step, phase, ctx, n_out, n_embd, "entry");
            std::vector<int32_t> pos_buf((size_t) n_out);
            std::vector<int32_t> tok_buf((size_t) n_out);
            for (int32_t i = 0; i < n_out; ++i) {
                pos_buf[(size_t) i] = req.pos_start + i;
                tok_buf[(size_t) i] = (i < (int32_t) tokens_i32.size()) ? tokens_i32[(size_t) i] : -1;
            }
            dist_debug_log_runtime_state(
                    dbg, debug_step, phase, "entry", ctx, n_out,
                    tok_buf.data(), pos_buf.data(), -1, in_tok);
        }

        split_gen3_a_resp resp{};
        std::vector<float> logits;
        if (!forward_to_peer(b_fd, ctx, n_out, n_embd, le, req.pos_start,
                req.include_logits != 0, n_vocab, ms_a, next_is_final,
                debug_step, phase, dbg, resp, logits)) {
            split_tcp_close(ctrl_fd);
            pipe_failed = true;
            break;
        }

        if (!split_gen3_send_a_resp(ctrl_fd, resp, logits.empty() ? nullptr : logits.data(), n_vocab)) {
            fprintf(stderr, "gen3_a: send resp failed\n");
            split_tcp_close(ctrl_fd);
            break;
        }

        if (dbg && resp.token_id >= 0) {
            dbg->emit_token_selected(debug_step, phase, resp.token_id, req.pos_start, false);
        }
        if (decode_step) {
            entry_queue_depth++;
        }
        debug_step++;
        }
        if (pipe_failed) {
            break;
        }
    }

    if (!normal_shutdown) {
        split_gen_write_ready_state(ready_file, pipe_failed ? "FAILED" : "STOPPED");
    }
    split_tcp_close(b_fd);
    split_tcp_close(listen_fd);

    llama_free(ctx);
    llama_model_free(model);
    return 0;
}
