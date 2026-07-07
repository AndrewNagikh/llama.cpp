// Process C: layers [layer_start, layer_end) or [layer_start, n_layer) - final stage

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

#include "llama-distributed.h"

#include <httplib.h>
#include <nlohmann/json.hpp>

#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

using json = nlohmann::json;

static void usage(const char * prog) {
    fprintf(stderr,
            "usage: %s MODEL --bc-port PORT [--layer-start N] [--layer-end N] "
            "[--external-output] [--output-http-host HOST] [--output-http-port PORT] "
            "[--ready-file PATH]\n",
            prog);
}

static bool output_service_sample_http(
        const std::string & http_host,
        int http_port,
        const float * hidden,
        int32_t n_embd,
        int32_t pos,
        int32_t & token_out,
        int32_t & n_vocab_out) {
    httplib::Client cli(http_host.c_str(), http_port);
    cli.set_connection_timeout(10, 0);
    cli.set_read_timeout(120, 0);
    json hidden_json = json::array();
    for (int32_t i = 0; i < n_embd; ++i) {
        hidden_json.push_back(hidden[i]);
    }
    const json body = {
        { "hidden", hidden_json },
        { "n_embd", n_embd },
        { "pos", pos },
    };
    const auto res = cli.Post("/runtime/output/sample", body.dump(), "application/json");
    if (!res || res->status != 200) {
        return false;
    }
    try {
        const json j = json::parse(res->body);
        if (!j.value("ok", false)) {
            return false;
        }
        token_out   = j.value("token_id", -1);
        n_vocab_out = j.value("n_vocab", 0);
        return token_out >= 0;
    } catch (...) {
        return false;
    }
}

static bool output_service_reset_http(const std::string & http_host, int http_port) {
    if (http_port <= 0) {
        return false;
    }
    httplib::Client cli(http_host.c_str(), http_port);
    cli.set_connection_timeout(10, 0);
    cli.set_read_timeout(60, 0);
    const auto res = cli.Post("/runtime/output/reset", "{}", "application/json");
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

static bool run_partial_layers(
        llama_context * ctx,
        const split_tcp_hidden_msg & msg,
        int32_t n_embd,
        int layer_start,
        int layer_end,
        double & ms_out,
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
                fprintf(stderr, "gen3_c: hidden API roundtrip FAIL: %s\n", api.message.c_str());
            }
        }
        if (split_gen_decode_hidden(ctx, msg.data.data(), n_tokens, n_embd, pos_start, true) != 0) {
            fprintf(stderr, "gen3_c: decode hidden failed n_tokens=%d\n", n_tokens);
            return false;
        }
        const float * h = llama_get_embeddings(ctx);
        if (h == nullptr) {
            fprintf(stderr, "gen3_c: missing embeddings after decode n_tokens=%d\n", n_tokens);
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
                    fprintf(stderr, "gen3_c: hidden API roundtrip FAIL i=%d: %s\n", i, api.message.c_str());
                }
            }
            if (split_gen_decode_hidden(ctx, in, 1, n_embd, pos_start + i, true) != 0) {
                fprintf(stderr, "gen3_c: decode hidden failed i=%d/%d\n", i, n_tokens);
                return false;
            }
            const float * h = llama_get_embeddings(ctx);
            if (h == nullptr) {
                fprintf(stderr, "gen3_c: missing embeddings i=%d/%d\n", i, n_tokens);
                return false;
            }
            std::memcpy(out_hidden.data() + (size_t) i * n_embd, h, (size_t) n_embd * sizeof(float));
        }
    }

    ms_out = (ggml_time_us() - t0) / 1000.0;
    return true;
}

int main(int argc, char ** argv) {
    split_tcp_init();
    if (argc < 2) {
        usage(argv[0]);
        return 1;
    }

    const char * model_path = argv[1];
    int bc_port     = -1;
    int layer_start = 8;
    int layer_end   = 0;
    std::string output_http_host = "127.0.0.1";
    int output_http_port = 0;
    bool external_output = false;
    const char * bind_host = "0.0.0.0";
    std::string ready_file;

    for (int i = 2; i < argc; ++i) {
        if (strcmp(argv[i], "--bc-port") == 0 && i + 1 < argc) {
            bc_port = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--bind") == 0 && i + 1 < argc) {
            bind_host = argv[++i];
        } else if (strcmp(argv[i], "--layer-start") == 0 && i + 1 < argc) {
            layer_start = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--layer-end") == 0 && i + 1 < argc) {
            layer_end = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--external-output") == 0) {
            external_output = true;
        } else if (strcmp(argv[i], "--output-http-host") == 0 && i + 1 < argc) {
            output_http_host = argv[++i];
        } else if (strcmp(argv[i], "--output-http-port") == 0 && i + 1 < argc) {
            output_http_port = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--ready-file") == 0 && i + 1 < argc) {
            ready_file = argv[++i];
        } else {
            usage(argv[0]);
            return 1;
        }
    }

    if (bc_port < 0) {
        usage(argv[0]);
        return 1;
    }

    split_gen_write_ready_state(ready_file, "MODEL_LOADING");

    const int listen_fd = split_tcp_listen_host(bind_host, bc_port);
    if (listen_fd < 0) {
        split_gen_write_ready_state(ready_file, "FAILED");
        fprintf(stderr, "gen3_c: listen failed %s:%d\n", bind_host, bc_port);
        return 1;
    }
    split_gen_write_ready_state(ready_file, "LISTENER_READY");

    ggml_backend_load_all();

    std::string load_err;
    llama_model * model = split_gen_load_model(model_path, load_err);
    if (!model) {
        split_gen_write_ready_state(ready_file, "FAILED");
        fprintf(stderr, "gen3_c: load model failed: %s\n", load_err.c_str());
        return 1;
    }

    const int32_t n_layer = llama_model_n_layer(model);
    const int32_t n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model));
    int32_t effective_layer_end = layer_end > layer_start ? layer_end : n_layer;
    if (external_output && effective_layer_end >= n_layer && layer_start + 1 < n_layer) {
        effective_layer_end = n_layer - 1;
    }

    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx   = 512;
    cparams.n_batch = 512;
    cparams.n_ubatch = 1;
    cparams.no_perf = true;
    cparams.layer_start = layer_start;
    cparams.layer_end   = effective_layer_end;
    cparams.skip_output_head = external_output;
    fprintf(stderr, "gen3_c: create context layer_start=%d layer_end=%d skip_output_head=%d\n",
            layer_start, effective_layer_end, external_output ? 1 : 0);
    llama_context * ctx = llama_init_from_model(model, cparams);
    if (!ctx) {
        split_gen_write_ready_state(ready_file, "FAILED");
        fprintf(stderr, "gen3_c: create context failed\n");
        return 1;
    }

    if (external_output) {
        llama_set_skip_output_head(ctx, true);
    }

    llama_set_layer_range(ctx, layer_start, effective_layer_end);
    llama_sampler * smpl = external_output ? nullptr : split_gen_make_sampler();

    dist_debug_load_config();
    dist_debug_reset_recorder("final");
    trace_recorder * dbg = dist_debug_recorder();
    int32_t debug_step   = 0;

    fprintf(stderr,
            "gen3_c: waiting bc_port=%d layer_start=%d layer_end=%d external_output=%d\n",
            bc_port, layer_start, effective_layer_end, external_output ? 1 : 0);
    split_gen_write_ready_state(ready_file, "PIPE_READY");
    split_gen_write_ready_state(ready_file, "READY");

    const int bc_fd = split_tcp_accept(listen_fd);
    split_tcp_close(listen_fd);
    if (bc_fd < 0) {
        fprintf(stderr, "gen3_c: accept failed\n");
        split_gen_write_ready_state(ready_file, "FAILED");
        return 1;
    }

    bool normal_shutdown = false;
    static int32_t final_queue_depth = 0;
    while (true) {
        split_ab_cmd cmd;
        perf_trace_sched_queue_wait_begin();
        const bool got_cmd = split_ab_recv_cmd(bc_fd, cmd);
        perf_trace_sched_queue_wait_end();
        if (!got_cmd) {
            fprintf(stderr, "gen3_c: recv cmd failed\n");
            break;
        }

        if (cmd == SPLIT_AB_CMD_SHUTDOWN) {
            normal_shutdown = true;
            break;
        }

        if (cmd == SPLIT_AB_CMD_RESET) {
            llama_memory_clear(llama_get_memory(ctx), true);
            llama_clear_hidden_state(ctx);
            if (external_output && !output_service_reset_http(output_http_host, output_http_port)) {
                fprintf(stderr, "gen3_c: output service reset failed\n");
                break;
            }
            debug_step = 0;
            continue;
        }

        if (cmd != SPLIT_AB_CMD_HIDDEN) {
            fprintf(stderr, "gen3_c: unexpected cmd %u\n", cmd);
            break;
        }

        split_tcp_hidden_msg msg;
        perf_trace_sched_queue_wait_begin();
        const bool got_hidden = split_ab_recv_hidden(bc_fd, msg);
        perf_trace_sched_queue_wait_end();
        if (!got_hidden) {
            fprintf(stderr, "gen3_c: recv hidden failed\n");
            break;
        }

        const char * phase = msg.header.n_tokens > 1 ? "prefill" : "decode";
        const int32_t tok_idx = (std::strcmp(phase, "decode") == 0) ? debug_step : -1;
        const bool decode_step = (std::strcmp(phase, "decode") == 0);
        perf_trace_refresh_context();
        if (decode_step && perf_trace_enabled()) {
            perf_trace_set_component("final");
            perf_emit_instant("FINAL_RECEIVE", perf_category::NETWORK, "final", tok_idx, nullptr);
            perf_emit_queue_depth("final", tok_idx, final_queue_depth);
        }
        if (decode_step) {
            final_queue_depth = std::max(0, final_queue_depth - 1);
        }

        hidden_transport_trace tr_recv = dist_debug_transport_recv(
                debug_step, phase, "bc",
                msg.header.n_tokens, msg.header.n_embd, msg.header.layer_end,
                msg.meta.pos_start, msg.data.data(), 0.0);
        if (dbg) {
            dbg->emit_transport(tr_recv);
        }
        if (!tr_recv.memcmp_ok && dist_debug_transport_dump_enabled()) {
            fprintf(stderr, "gen3_c: TCP hidden memcmp FAIL step=%d link=bc\n", debug_step);
        }

        if (dbg) {
            dbg->emit_step_begin(debug_step, phase, -1, msg.meta.pos_start, 0);
            dist_debug_log_position(dbg, debug_step, phase, ctx, msg.meta.pos_start,
                    msg.header.n_tokens, 1);
        }

        const int32_t n_emb = msg.header.n_embd > 0 ? msg.header.n_embd : llama_model_n_embd(model);
        const int32_t n_tok = msg.header.n_tokens > 0 ? msg.header.n_tokens : 1;
        const int32_t sample_idx = n_tok - 1;
        const int32_t sample_pos = msg.meta.pos_start + sample_idx;

        double ms_compute = 0.0;
        int32_t token_id = -1;
        int32_t resp_vocab = n_vocab;
        double ms_sample = 0.0;

        if (external_output) {
            std::vector<float> out_hidden;
            split_gen_pipe_trace("final", phase, "partial_decode_enter",
                    msg.header.n_tokens, n_emb, msg.meta.pos_start, layer_start, effective_layer_end);
            if (!run_partial_layers(
                        ctx, msg, n_emb, layer_start, effective_layer_end, ms_compute, out_hidden)) {
                fprintf(stderr, "gen3_c: partial forward failed\n");
                break;
            }
            split_gen_pipe_trace("final", phase, "partial_decode_exit",
                    msg.header.n_tokens, n_emb, msg.meta.pos_start, layer_start, effective_layer_end);
            const float * sample_hidden = out_hidden.data() + (size_t) sample_idx * (size_t) n_emb;
            const int64_t t1 = ggml_time_us();
            perf_span sample_span("SAMPLER_BEGIN", "SAMPLER_END", perf_category::SAMPLING, "final");
            sample_span.set_token_idx(tok_idx);
            if (output_http_port <= 0 ||
                    !output_service_sample_http(
                            output_http_host, output_http_port,
                            sample_hidden, n_emb, sample_pos, token_id, resp_vocab)) {
                fprintf(stderr, "gen3_c: output service sample failed\n");
                break;
            }
            ms_sample = (ggml_time_us() - t1) / 1000.0;
        } else {
            llama_set_layer_range(ctx, layer_start, n_layer);
            llama_set_hidden_state(ctx, msg.data.data(), msg.header.n_tokens);
            {
                const hidden_state_api_check api = verify_hidden_state_roundtrip(
                        ctx, msg.data.data(), msg.header.n_tokens, msg.header.n_embd);
                if (!api.ok && dist_debug_transport_dump_enabled()) {
                    fprintf(stderr, "gen3_c: hidden API roundtrip FAIL: %s\n", api.message.c_str());
                }
            }

            const int64_t t0 = ggml_time_us();
            split_gen_pipe_trace("final", phase, "decode_enter",
                    msg.header.n_tokens, msg.header.n_embd, msg.meta.pos_start, layer_start, n_layer);
            perf_span compute_span("FINAL_COMPUTE_BEGIN", "FINAL_COMPUTE_END", perf_category::COMPUTE, "final");
            compute_span.set_token_idx(tok_idx);
            if (split_gen_decode_hidden(ctx, msg.data.data(), msg.header.n_tokens, msg.header.n_embd,
                        msg.meta.pos_start, true) != 0) {
                fprintf(stderr, "gen3_c: decode failed\n");
                break;
            }
            split_gen_pipe_trace("final", phase, "decode_exit",
                    msg.header.n_tokens, msg.header.n_embd, msg.meta.pos_start, layer_start, n_layer);
            ms_compute = (ggml_time_us() - t0) / 1000.0;

            if (dbg) {
                dist_debug_log_kv(dbg, debug_step, phase, ctx, 0);
                dist_debug_log_hidden_out(dbg, debug_step, phase, ctx, msg.header.n_tokens, n_emb, "final");
                dist_debug_log_logits_out(dbg, debug_step, phase, ctx, n_vocab, -1);
            }

            const int64_t t1 = ggml_time_us();
            perf_span sample_span("SAMPLER_BEGIN", "SAMPLER_END", perf_category::SAMPLING, "final");
            sample_span.set_token_idx(tok_idx);
            bool used_argmax = false;
            const llama_token sampled = static_cast<llama_token>(
                    dist_debug_sample_or_argmax(smpl, ctx, -1, &used_argmax));
            if (!used_argmax) {
                llama_sampler_accept(smpl, sampled);
            }
            token_id = (int32_t) sampled;
            ms_sample = (ggml_time_us() - t1) / 1000.0;

            if (dbg) {
                dbg->emit_token_selected(debug_step, phase, token_id, msg.meta.pos_start, used_argmax);
            }
        }

        if (dbg) {
            dist_debug_log_runtime_state(
                    dbg, debug_step, phase, "final", ctx, msg.header.n_tokens,
                    nullptr, nullptr, -1, token_id);
        }

        split_gen3_c_resp resp{};
        resp.magic      = SPLIT_GEN_MAGIC;
        resp.token_id   = token_id;
        resp.n_vocab    = resp_vocab;
        resp.ms_compute = ms_compute;
        resp.ms_sample  = ms_sample;

        if (!split_gen3_send_c_resp(bc_fd, resp)) {
            fprintf(stderr, "gen3_c: send resp failed\n");
            break;
        }
        if (decode_step) {
            final_queue_depth++;
        }
        debug_step++;
    }

    split_gen_write_ready_state(ready_file, normal_shutdown ? "STOPPED" : "FAILED");
    split_tcp_close(bc_fd);

    if (smpl) {
        llama_sampler_free(smpl);
    }
    llama_free(ctx);
    llama_model_free(model);
    return 0;
}
