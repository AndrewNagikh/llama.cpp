// Process B: middle stage - layers [layer_start, layer_end)

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
#include "wave_inbound_queue.h"

#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <algorithm>
#include <deque>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

static void usage(const char * prog) {
    fprintf(stderr,
            "usage: %s MODEL --ab-port PORT --bc-port PORT [--layer-start N] "
            "[--layer-end N] [--ready-file PATH]\n",
            prog);
}

static bool forward_to_c(
        int bc_fd,
        int32_t n_tokens,
        int32_t n_embd,
        int32_t layer_end,
        int32_t pos_start,
        double ms_b,
        const float * hidden,
        int32_t debug_step,
        const char * phase,
        trace_recorder * dbg,
        split_gen3_mid_resp & resp) {
    if (hidden == nullptr && n_tokens > 0) {
        fprintf(stderr, "gen3_b: missing hidden state\n");
        return false;
    }

    hidden_transport_trace tr_send = dist_debug_transport_send(
            debug_step, phase, "bc", n_tokens, n_embd, layer_end, pos_start, hidden, 0.0);
    if (dbg) {
        dbg->emit_transport(tr_send);
    }

    const int64_t t0 = ggml_time_us();
    if (!split_ab_send_hidden(bc_fd, n_tokens, n_embd, layer_end, pos_start, 0, hidden)) {
        fprintf(stderr, "gen3_b: send hidden to C failed\n");
        return false;
    }
    const int64_t send_us = ggml_time_us() - t0;

    split_gen3_c_resp cresp{};
    if (!split_gen3_recv_c_resp(bc_fd, cresp)) {
        fprintf(stderr, "gen3_b: recv C resp failed\n");
        return false;
    }

    if (perf_trace_enabled()) {
        const int32_t wave_id = perf_trace_wave_id_from_step(phase, debug_step);
        const int32_t tok_idx = (phase && std::strcmp(phase, "decode") == 0) ? debug_step : -1;
        if (phase && std::strcmp(phase, "decode") == 0) {
            perf_trace_ensure_decode_context(tok_idx, wave_id);
        } else {
            perf_trace_set_wave_id(wave_id);
        }
        const int32_t payload_bytes = n_tokens * n_embd * (int32_t) sizeof(float);
        perf_trace_set_component("middle");
        perf_emit_hidden_transfer("middle", tok_idx, "bc", payload_bytes, 0, send_us, 0, 0);
        if (phase && std::strcmp(phase, "decode") == 0) {
            perf_emit_span("MIDDLE_SEND_END", perf_category::NETWORK, "middle", tok_idx, send_us, nullptr);
        }
    }

    resp.magic       = SPLIT_GEN_MAGIC;
    resp.token_id    = cresp.token_id;
    resp.n_vocab     = cresp.n_vocab;
    resp.ms_b_compute = ms_b;
    resp.ms_bc_xfer   = send_us / 1000.0;
    resp.ms_c_compute = cresp.ms_compute;
    resp.ms_c_sample  = cresp.ms_sample;
    return true;
}

static bool run_b_layers(
        llama_context * ctx,
        const split_tcp_hidden_msg & msg,
        int32_t n_embd,
        int layer_start,
        int layer_end,
        double & ms_b_out,
        std::vector<float> & out_hidden);

static bool send_hidden_to_c_only(
        int bc_fd,
        int32_t n_tokens,
        int32_t n_embd,
        int32_t layer_end,
        int32_t pos_start,
        const float * hidden,
        int32_t debug_step,
        const char * phase,
        trace_recorder * dbg,
        double & send_ms_out) {
    if (hidden == nullptr && n_tokens > 0) {
        fprintf(stderr, "gen3_b: missing hidden state\n");
        return false;
    }

    hidden_transport_trace tr_send = dist_debug_transport_send(
            debug_step, phase, "bc", n_tokens, n_embd, layer_end, pos_start, hidden, 0.0);
    if (dbg) {
        dbg->emit_transport(tr_send);
    }

    const int64_t t0 = ggml_time_us();
    if (!split_ab_send_hidden(bc_fd, n_tokens, n_embd, layer_end, pos_start, 0, hidden)) {
        fprintf(stderr, "gen3_b: send hidden to C failed\n");
        return false;
    }
    const int64_t send_us = ggml_time_us() - t0;
    send_ms_out = send_us / 1000.0;

    if (perf_trace_enabled()) {
        const int32_t wave_id = perf_trace_wave_id_from_step(phase, debug_step);
        const int32_t tok_idx = (phase && std::strcmp(phase, "decode") == 0) ? debug_step : -1;
        if (phase && std::strcmp(phase, "decode") == 0) {
            perf_trace_ensure_decode_context(tok_idx, wave_id);
        } else {
            perf_trace_set_wave_id(wave_id);
        }
        const int32_t payload_bytes = n_tokens * n_embd * (int32_t) sizeof(float);
        perf_trace_set_component("middle");
        perf_emit_hidden_transfer("middle", tok_idx, "bc", payload_bytes, 0, send_us, 0, 0);
        if (phase && std::strcmp(phase, "decode") == 0) {
            perf_emit_span("MIDDLE_SEND_END", perf_category::NETWORK, "middle", tok_idx, send_us, nullptr);
        }
    }
    return true;
}

struct middle_stage_state {
    llama_context * ctx         = nullptr;
    int             ab_fd       = -1;
    int             bc_fd       = -1;
    int             layer_start = 0;
    int             layer_end   = 0;
    int32_t         n_embd      = 0;
    trace_recorder * dbg        = nullptr;
    int32_t *       debug_step  = nullptr;
    int32_t *       queue_depth = nullptr;
};

struct middle_bc_pending {
    int32_t debug_step = 0;
    int32_t wave_id    = -1;
    bool    decode_step = false;
    double  ms_b        = 0.0;
    double  ms_bc_xfer  = 0.0;
    int32_t pos_start   = 0;
};

static bool middle_process_hidden_item(
        const middle_stage_state & st,
        hidden_wave_work_item & item,
        const bool pipeline_bc,
        std::mutex * bc_mu,
        std::deque<middle_bc_pending> * bc_pending) {
    split_tcp_hidden_msg & msg = item.msg;
    const char * phase = msg.header.n_tokens > 1 ? "prefill" : "decode";
    const int32_t step = item.debug_step;
    const int32_t wave_id = item.wave_id >= 0
            ? item.wave_id
            : perf_trace_wave_id_from_step(phase, step);
    const int32_t tok_idx = (std::strcmp(phase, "decode") == 0) ? step : -1;
    const bool decode_step = (std::strcmp(phase, "decode") == 0);

    if (decode_step && perf_trace_enabled()) {
        perf_trace_ensure_decode_context(tok_idx, wave_id);
        perf_trace_set_component("middle");
        perf_emit_instant("MIDDLE_RECEIVE", perf_category::NETWORK, "middle", tok_idx, nullptr);
        if (st.queue_depth) {
            perf_emit_queue_depth("middle", tok_idx, *st.queue_depth);
            *st.queue_depth = std::max(0, *st.queue_depth - 1);
        }
    }

    hidden_transport_trace tr_recv = dist_debug_transport_recv(
            step, phase, "ab",
            msg.header.n_tokens, msg.header.n_embd, msg.header.layer_end,
            msg.meta.pos_start, msg.data.data(), 0.0);
    if (st.dbg) {
        st.dbg->emit_transport(tr_recv);
        st.dbg->emit_step_begin(step, phase, -1, msg.meta.pos_start, 0);
        dist_debug_log_position(st.dbg, step, phase, st.ctx, msg.meta.pos_start,
                msg.header.n_tokens, 1);
    }
    if (!tr_recv.memcmp_ok && dist_debug_transport_dump_enabled()) {
        fprintf(stderr, "gen3_b: TCP hidden memcmp FAIL step=%d link=ab\n", step);
    }

    llama_set_layer_range(st.ctx, st.layer_start, st.layer_end);

    double ms_b = 0.0;
    std::vector<float> out_hidden;
    split_gen_pipe_trace("middle", phase, "decode_enter",
            msg.header.n_tokens, st.n_embd, msg.meta.pos_start, st.layer_start, st.layer_end);
    if (perf_trace_enabled() && decode_step) {
        perf_trace_ensure_decode_context(tok_idx, wave_id);
    }
    perf_span compute_span("MIDDLE_COMPUTE_BEGIN", "MIDDLE_COMPUTE_END", perf_category::COMPUTE, "middle");
    compute_span.set_token_idx(tok_idx);
    if (!run_b_layers(st.ctx, msg, st.n_embd, st.layer_start, st.layer_end, ms_b, out_hidden)) {
        fprintf(stderr, "gen3_b: decode failed\n");
        return false;
    }
    split_gen_pipe_trace("middle", phase, "decode_exit",
            msg.header.n_tokens, st.n_embd, msg.meta.pos_start, st.layer_start, st.layer_end);

    if (st.dbg) {
        st.dbg->emit_hidden(step, phase, out_hidden.data(), msg.header.n_tokens, st.n_embd, "middle");
        dist_debug_log_kv(st.dbg, step, phase, st.ctx, 0);
    }

    split_gen_pipe_trace("middle", phase, "forward_to_final_enter",
            msg.header.n_tokens, st.n_embd, msg.meta.pos_start, st.layer_start, st.layer_end);

    if (pipeline_bc && bc_mu != nullptr && bc_pending != nullptr) {
        middle_bc_pending pending;
        pending.debug_step  = step;
        pending.wave_id     = wave_id;
        pending.decode_step = decode_step;
        pending.ms_b        = ms_b;
        pending.ms_bc_xfer  = 0.0;
        pending.pos_start   = msg.meta.pos_start;
        {
            std::lock_guard<std::mutex> lock(*bc_mu);
            bc_pending->push_back(pending);
        }
        double send_ms = 0.0;
        if (!send_hidden_to_c_only(st.bc_fd, msg.header.n_tokens, st.n_embd, st.layer_end,
                    msg.meta.pos_start, out_hidden.data(), step, phase, st.dbg, send_ms)) {
            std::lock_guard<std::mutex> lock(*bc_mu);
            if (!bc_pending->empty()) {
                bc_pending->pop_back();
            }
            return false;
        }
        {
            std::lock_guard<std::mutex> lock(*bc_mu);
            if (!bc_pending->empty()) {
                bc_pending->back().ms_bc_xfer = send_ms;
            }
        }
    } else {
        split_gen3_mid_resp resp{};
        if (!forward_to_c(st.bc_fd, msg.header.n_tokens, st.n_embd, st.layer_end,
                    msg.meta.pos_start, ms_b, out_hidden.data(), step, phase, st.dbg, resp)) {
            return false;
        }
        split_gen_pipe_trace("middle", phase, "forward_to_final_exit",
                msg.header.n_tokens, st.n_embd, msg.meta.pos_start, st.layer_start, st.layer_end);
        if (st.dbg) {
            dist_debug_log_runtime_state(
                    st.dbg, step, phase, "middle", st.ctx, msg.header.n_tokens,
                    nullptr, nullptr, -1, resp.token_id);
            if (resp.token_id >= 0) {
                st.dbg->emit_token_selected(step, phase, resp.token_id, msg.meta.pos_start, false);
            }
        }
        if (!split_gen3_send_mid_resp(st.ab_fd, resp)) {
            fprintf(stderr, "gen3_b: send mid resp failed\n");
            return false;
        }
    }

    if (decode_step && st.queue_depth) {
        (*st.queue_depth)++;
    }
    if (st.debug_step) {
        (*st.debug_step)++;
    }
    return true;
}

static bool middle_run_queued_session(
        middle_stage_state st,
        hidden_inbound_queue & queue,
        int32_t & debug_step,
        bool & normal_shutdown,
        bool & pipe_failed) {
    std::mutex ab_mu;
    std::mutex bc_mu;
    std::deque<middle_bc_pending> bc_pending;
    std::atomic<bool> stop{ false };
    int32_t middle_queue_depth = 0;
    st.debug_step = &debug_step;
    st.queue_depth = &middle_queue_depth;

    // RESET arrives on the receiver thread's control-channel recv loop, while
    // the consumer loop below concurrently drives decode on the same st.ctx
    // from queued hidden-state items. Without this lock the two threads
    // mutate the same llama_context (KV cache clear vs. active decode) at
    // once, corrupting ggml's allocator state.
    std::mutex ctx_mu;

    std::thread receiver([&] {
        while (!stop.load()) {
            split_ab_cmd cmd;
            perf_trace_sched_queue_wait_begin();
            const bool got_cmd = split_ab_recv_cmd(st.ab_fd, cmd);
            perf_trace_sched_queue_wait_end();
            if (!got_cmd) {
                stop.store(true);
                break;
            }

            if (cmd == SPLIT_AB_CMD_SHUTDOWN) {
                split_ab_send_shutdown(st.bc_fd);
                normal_shutdown = true;
                stop.store(true);
                break;
            }

            if (cmd == SPLIT_AB_CMD_RESET) {
                {
                    std::lock_guard<std::mutex> lock(ctx_mu);
                    llama_memory_clear(llama_get_memory(st.ctx), true);
                    llama_clear_hidden_state(st.ctx);
                }
                debug_step = 0;
                middle_queue_depth = 0;
                split_ab_send_reset(st.bc_fd);
                continue;
            }

            if (cmd != SPLIT_AB_CMD_HIDDEN) {
                fprintf(stderr, "gen3_b: unexpected cmd %u\n", cmd);
                stop.store(true);
                break;
            }

            split_tcp_hidden_msg msg;
            perf_trace_sched_queue_wait_begin();
            const bool got_hidden = split_ab_recv_hidden(st.ab_fd, msg);
            perf_trace_sched_queue_wait_end();
            if (!got_hidden) {
                stop.store(true);
                break;
            }

            const char * phase = msg.header.n_tokens > 1 ? "prefill" : "decode";
            const int32_t queued_wave_id = perf_trace_wave_id_from_step(phase, debug_step);
            hidden_wave_work_item item;
            item.msg        = std::move(msg);
            item.debug_step = debug_step;
            item.wave_id    = queued_wave_id;

            queue.push(std::move(item));
            middle_queue_depth = queue.observable_depth(true);
            if (perf_trace_enabled()) {
                perf_trace_set_wave_id(queued_wave_id);
                perf_emit_instant("WAVE_QUEUED", perf_category::WAIT, "middle", -1, nullptr);
                perf_emit_queue_depth("middle", -1, middle_queue_depth);
            }
        }
        // Wake a consumer parked in queue.pop(); nothing else ever will
        // once this producer thread has exited (see split_gen3_a.cpp).
        queue.close();
    });

    std::thread bc_responder([&] {
        while (!stop.load() || !bc_pending.empty()) {
            middle_bc_pending pending;
            {
                std::unique_lock<std::mutex> lock(bc_mu);
                while (!stop.load() && bc_pending.empty()) {
                    lock.unlock();
                    std::this_thread::sleep_for(std::chrono::milliseconds(1));
                    lock.lock();
                }
                if (bc_pending.empty()) {
                    break;
                }
                pending = bc_pending.front();
                bc_pending.pop_front();
            }

            split_gen3_c_resp cresp{};
            if (!split_gen3_recv_c_resp(st.bc_fd, cresp)) {
                if (!stop.load()) {
                    pipe_failed = true;
                }
                stop.store(true);
                break;
            }

            split_gen3_mid_resp resp{};
            resp.magic        = SPLIT_GEN_MAGIC;
            resp.token_id     = cresp.token_id;
            resp.n_vocab      = cresp.n_vocab;
            resp.ms_b_compute = pending.ms_b;
            resp.ms_bc_xfer   = pending.ms_bc_xfer;
            resp.ms_c_compute = cresp.ms_compute;
            resp.ms_c_sample  = cresp.ms_sample;

            const char * phase = pending.decode_step ? "decode" : "prefill";
            if (st.dbg) {
                dist_debug_log_runtime_state(
                        st.dbg, pending.debug_step, phase, "middle", st.ctx, 1,
                        nullptr, nullptr, -1, resp.token_id);
                if (resp.token_id >= 0) {
                    st.dbg->emit_token_selected(
                            pending.debug_step, phase, resp.token_id, pending.pos_start, false);
                }
            }

            std::lock_guard<std::mutex> lock(ab_mu);
            if (!split_gen3_send_mid_resp(st.ab_fd, resp)) {
                fprintf(stderr, "gen3_b: send mid resp failed\n");
                pipe_failed = true;
                stop.store(true);
                break;
            }
        }
    });

    while (!stop.load() || queue.depth() > 0) {
        hidden_wave_work_item item;
        if (!queue.try_pop(item)) {
            if (stop.load()) {
                break;
            }
            if (!queue.pop(item)) {
                // Queue closed by the receiver and fully drained.
                break;
            }
        }
        bool ok;
        {
            std::lock_guard<std::mutex> lock(ctx_mu);
            ok = middle_process_hidden_item(st, item, true, &bc_mu, &bc_pending);
        }
        if (!ok) {
            pipe_failed = true;
            stop.store(true);
            break;
        }
    }

    stop.store(true);
    // Wake a receiver parked in a blocking push() so the join can complete.
    queue.close();
    if (receiver.joinable()) {
        receiver.join();
    }
    if (bc_responder.joinable()) {
        bc_responder.join();
    }
    return !pipe_failed;
}

static bool run_b_layers(
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
        {
            const hidden_state_api_check api = verify_hidden_state_roundtrip(
                    ctx, msg.data.data(), n_tokens, n_embd);
            if (!api.ok && dist_debug_transport_dump_enabled()) {
                fprintf(stderr, "gen3_b: hidden API roundtrip FAIL: %s\n", api.message.c_str());
            }
        }
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
            {
                const hidden_state_api_check api = verify_hidden_state_roundtrip(ctx, in, 1, n_embd);
                if (!api.ok && dist_debug_transport_dump_enabled()) {
                    fprintf(stderr, "gen3_b: hidden API roundtrip FAIL i=%d: %s\n", i, api.message.c_str());
                }
            }
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

int main(int argc, char ** argv) {
    split_tcp_init();
    if (argc < 2) {
        usage(argv[0]);
        return 1;
    }

    const char * model_path = argv[1];
    int ab_port     = -1;
    int bc_port     = -1;
    int layer_start = 5;
    int layer_end   = 10;
    const char * bc_host = "127.0.0.1";
    const char * bind_host = "0.0.0.0";
    std::string ready_file;

    for (int i = 2; i < argc; ++i) {
        if (strcmp(argv[i], "--ab-port") == 0 && i + 1 < argc) {
            ab_port = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--bc-port") == 0 && i + 1 < argc) {
            bc_port = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--bc-host") == 0 && i + 1 < argc) {
            bc_host = argv[++i];
        } else if (strcmp(argv[i], "--bind") == 0 && i + 1 < argc) {
            bind_host = argv[++i];
        } else if (strcmp(argv[i], "--layer-start") == 0 && i + 1 < argc) {
            layer_start = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--layer-end") == 0 && i + 1 < argc) {
            layer_end = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--ready-file") == 0 && i + 1 < argc) {
            ready_file = argv[++i];
        } else {
            usage(argv[0]);
            return 1;
        }
    }

    if (ab_port < 0 || bc_port < 0) {
        usage(argv[0]);
        return 1;
    }

    split_gen_write_ready_state(ready_file, "MODEL_LOADING");

    const int listen_fd = split_tcp_listen_host(bind_host, ab_port);
    if (listen_fd < 0) {
        split_gen_write_ready_state(ready_file, "FAILED");
        fprintf(stderr, "gen3_b: listen failed %s:%d\n", bind_host, ab_port);
        return 1;
    }
    split_gen_write_ready_state(ready_file, "LISTENER_READY");

    ggml_backend_load_all();

    std::string load_err;
    llama_model * model = split_gen_load_model(model_path, load_err);
    if (!model) {
        split_gen_write_ready_state(ready_file, "FAILED");
        fprintf(stderr, "gen3_b: load model failed: %s\n", load_err.c_str());
        return 1;
    }

    const int32_t n_embd = llama_model_n_embd(model);

    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx   = 512;
    cparams.n_batch = 512;
    cparams.n_ubatch = 1;
    cparams.no_perf = true;
    cparams.layer_start = layer_start;
    cparams.layer_end   = layer_end;
    fprintf(stderr, "gen3_b: create context layer_start=%d layer_end=%d\n", layer_start, layer_end);
    llama_context * ctx = llama_init_from_model(model, cparams);
    if (!ctx) {
        split_gen_write_ready_state(ready_file, "FAILED");
        fprintf(stderr, "gen3_b: create context failed\n");
        return 1;
    }

    llama_set_layer_range(ctx, layer_start, layer_end);

    dist_debug_load_config();
    dist_debug_reset_recorder("middle");
    trace_recorder * dbg = dist_debug_recorder();
    int32_t debug_step   = 0;

    const int bc_fd = split_tcp_connect_retry(bc_host, bc_port, 3000, 100);
    if (bc_fd < 0) {
        split_gen_write_ready_state(ready_file, "FAILED");
        fprintf(stderr, "gen3_b: connect to C failed %s:%d\n", bc_host, bc_port);
        return 1;
    }
    split_gen_write_ready_state(ready_file, "PIPE_READY");

    fprintf(stderr, "gen3_b: ready ab_port=%d layer=[%d,%d)\n", ab_port, layer_start, layer_end);
    split_gen_write_ready_state(ready_file, "READY");

    const int ab_fd = split_tcp_accept(listen_fd);
    split_tcp_close(listen_fd);
    if (ab_fd < 0) {
        fprintf(stderr, "gen3_b: accept failed\n");
        split_gen_write_ready_state(ready_file, "FAILED");
        return 1;
    }

    bool normal_shutdown = false;
    bool pipe_failed = false;
    int32_t queue_depth = 0;

    if (runtime_stage_queue_enabled()) {
        hidden_inbound_queue queue(runtime_stage_queue_max_depth());
        middle_stage_state st{
            ctx, ab_fd, bc_fd, layer_start, layer_end, n_embd, dbg, &debug_step, &queue_depth
        };
        if (!middle_run_queued_session(st, queue, debug_step, normal_shutdown, pipe_failed)) {
            split_gen_write_ready_state(ready_file, "FAILED");
        } else {
            split_gen_write_ready_state(ready_file, normal_shutdown ? "STOPPED" : "FAILED");
        }
        split_tcp_close(ab_fd);
        split_tcp_close(bc_fd);
        llama_free(ctx);
        llama_model_free(model);
        return pipe_failed ? 1 : 0;
    }

    while (true) {
        split_ab_cmd cmd;
        perf_trace_sched_queue_wait_begin();
        const bool got_cmd = split_ab_recv_cmd(ab_fd, cmd);
        perf_trace_sched_queue_wait_end();
        if (!got_cmd) {
            fprintf(stderr, "gen3_b: recv cmd failed\n");
            break;
        }

        if (cmd == SPLIT_AB_CMD_SHUTDOWN) {
            split_ab_send_shutdown(bc_fd);
            normal_shutdown = true;
            break;
        }

        if (cmd == SPLIT_AB_CMD_RESET) {
            llama_memory_clear(llama_get_memory(ctx), true);
            llama_clear_hidden_state(ctx);
            debug_step = 0;
            split_ab_send_reset(bc_fd);
            continue;
        }

        if (cmd != SPLIT_AB_CMD_HIDDEN) {
            fprintf(stderr, "gen3_b: unexpected cmd %u\n", cmd);
            break;
        }

        split_tcp_hidden_msg msg;
        perf_trace_sched_queue_wait_begin();
        const bool got_hidden = split_ab_recv_hidden(ab_fd, msg);
        perf_trace_sched_queue_wait_end();
        if (!got_hidden) {
            fprintf(stderr, "gen3_b: recv hidden failed\n");
            break;
        }

        const char * phase = msg.header.n_tokens > 1 ? "prefill" : "decode";
        const int32_t tok_idx = (std::strcmp(phase, "decode") == 0) ? debug_step : -1;
        const bool decode_step = (std::strcmp(phase, "decode") == 0);
        if (decode_step && perf_trace_enabled()) {
            perf_trace_ensure_decode_context(tok_idx, perf_trace_wave_id_from_step(phase, debug_step));
            perf_trace_set_component("middle");
            perf_emit_instant("MIDDLE_RECEIVE", perf_category::NETWORK, "middle", tok_idx, nullptr);
            perf_emit_queue_depth("middle", tok_idx, queue_depth);
        }
        if (decode_step) {
            queue_depth = std::max(0, queue_depth - 1);
        }

        hidden_transport_trace tr_recv = dist_debug_transport_recv(
                debug_step, phase, "ab",
                msg.header.n_tokens, msg.header.n_embd, msg.header.layer_end,
                msg.meta.pos_start, msg.data.data(), 0.0);
        if (dbg) {
            dbg->emit_transport(tr_recv);
        }
        if (!tr_recv.memcmp_ok && dist_debug_transport_dump_enabled()) {
            fprintf(stderr, "gen3_b: TCP hidden memcmp FAIL step=%d link=ab\n", debug_step);
        }

        if (dbg) {
            dbg->emit_step_begin(debug_step, phase, -1, msg.meta.pos_start, 0);
            dist_debug_log_position(dbg, debug_step, phase, ctx, msg.meta.pos_start,
                    msg.header.n_tokens, 1);
        }

        llama_set_layer_range(ctx, layer_start, layer_end);

        double ms_b = 0.0;
        std::vector<float> out_hidden;
        split_gen_pipe_trace("middle", phase, "decode_enter",
                msg.header.n_tokens, n_embd, msg.meta.pos_start, layer_start, layer_end);
        if (perf_trace_enabled() && decode_step) {
            perf_trace_ensure_decode_context(
                    tok_idx, perf_trace_wave_id_from_step(phase, debug_step));
        }
        perf_span compute_span("MIDDLE_COMPUTE_BEGIN", "MIDDLE_COMPUTE_END", perf_category::COMPUTE, "middle");
        compute_span.set_token_idx(tok_idx);
        if (!run_b_layers(ctx, msg, n_embd, layer_start, layer_end, ms_b, out_hidden)) {
            fprintf(stderr, "gen3_b: decode failed\n");
            break;
        }
        split_gen_pipe_trace("middle", phase, "decode_exit",
                msg.header.n_tokens, n_embd, msg.meta.pos_start, layer_start, layer_end);

        if (dbg) {
            dist_debug_log_kv(dbg, debug_step, phase, ctx, 0);
            if (!out_hidden.empty()) {
                dbg->emit_hidden(debug_step, phase, out_hidden.data(),
                        msg.header.n_tokens, n_embd, "middle");
            }
        }

        split_gen3_mid_resp resp{};
        split_gen_pipe_trace("middle", phase, "forward_to_final_enter",
                msg.header.n_tokens, n_embd, msg.meta.pos_start, layer_start, layer_end);
        if (!forward_to_c(bc_fd, msg.header.n_tokens, n_embd, layer_end,
                msg.meta.pos_start, ms_b, out_hidden.data(),
                debug_step, phase, dbg, resp)) {
            break;
        }
        split_gen_pipe_trace("middle", phase, "forward_to_final_exit",
                msg.header.n_tokens, n_embd, msg.meta.pos_start, layer_start, layer_end);

        if (dbg) {
            dist_debug_log_runtime_state(
                    dbg, debug_step, phase, "middle", ctx, msg.header.n_tokens,
                    nullptr, nullptr, -1, resp.token_id);
        }

        if (!split_gen3_send_mid_resp(ab_fd, resp)) {
            fprintf(stderr, "gen3_b: send mid resp failed\n");
            break;
        }
        if (dbg && resp.token_id >= 0) {
            dbg->emit_token_selected(debug_step, phase, resp.token_id, msg.meta.pos_start, false);
        }
        if (decode_step) {
            queue_depth++;
        }
        debug_step++;
    }

    split_gen_write_ready_state(ready_file, normal_shutdown ? "STOPPED" : "FAILED");
    split_tcp_close(ab_fd);
    split_tcp_close(bc_fd);

    llama_free(ctx);
    llama_model_free(model);
    return 0;
}
