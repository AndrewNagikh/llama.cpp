// Process A: entry stage - layers [0, layer_end)

#include "ggml-backend.h"
#include "ggml.h"
#include "llama.h"
#include "split_gen_common.h"
#include "split_tcp_wire.h"

#include <cstdio>
#include <cstring>
#include <vector>

#if !defined(_WIN32)
#include <unistd.h>
#endif

static void usage(const char * prog) {
    fprintf(stderr, "usage: %s MODEL --ctrl-port PORT --b-port PORT [--layer-end N]\n", prog);
}

static bool forward_to_b(
        int b_fd,
        llama_context * ctx,
        int32_t n_tokens,
        int32_t n_embd,
        int32_t layer_end,
        int32_t pos_start,
        bool include_logits,
        int32_t n_vocab,
        double ms_a,
        split_gen3_a_resp & resp,
        std::vector<float> & logits_out) {
    const float * hidden = llama_get_embeddings(ctx);
    if (hidden == nullptr && n_tokens > 0) {
        fprintf(stderr, "gen3_a: missing hidden state\n");
        return false;
    }

    const int64_t t0 = ggml_time_us();
    if (!split_ab_send_hidden(b_fd, n_tokens, n_embd, layer_end, pos_start,
            include_logits ? 1 : 0, hidden)) {
        fprintf(stderr, "gen3_a: send hidden failed\n");
        return false;
    }
    const int64_t t1 = ggml_time_us();

    split_gen3_mid_resp mid{};
    if (!split_gen3_recv_mid_resp(b_fd, mid)) {
        fprintf(stderr, "gen3_a: recv mid resp failed\n");
        return false;
    }

    resp.magic          = SPLIT_GEN_MAGIC;
    resp.version        = SPLIT_GEN3_VERSION;
    resp.token_id       = mid.token_id;
    resp.include_logits = include_logits ? 1 : 0;
    resp.n_vocab        = n_vocab;
    resp.ms_a_compute   = ms_a;
    resp.ms_ab_xfer     = (t1 - t0) / 1000.0;
    resp.ms_b_compute   = mid.ms_b_compute;
    resp.ms_bc_xfer     = mid.ms_bc_xfer;
    resp.ms_c_compute   = mid.ms_c_compute;
    resp.ms_c_sample    = mid.ms_c_sample;
    logits_out.clear();
    return true;
}

int main(int argc, char ** argv) {
    if (argc < 2) {
        usage(argv[0]);
        return 1;
    }

    const char * model_path = argv[1];
    int ctrl_port = -1;
    int b_port    = -1;
    int layer_end = 5;
    const char * b_host = "127.0.0.1";
    const char * bind_host = "0.0.0.0";

    for (int i = 2; i < argc; ++i) {
        if (strcmp(argv[i], "--ctrl-port") == 0 && i + 1 < argc) {
            ctrl_port = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--b-port") == 0 && i + 1 < argc) {
            b_port = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--b-host") == 0 && i + 1 < argc) {
            b_host = argv[++i];
        } else if (strcmp(argv[i], "--bind") == 0 && i + 1 < argc) {
            bind_host = argv[++i];
        } else if (strcmp(argv[i], "--layer-end") == 0 && i + 1 < argc) {
            layer_end = atoi(argv[++i]);
        } else {
            usage(argv[0]);
            return 1;
        }
    }

    if (ctrl_port < 0 || b_port < 0) {
        usage(argv[0]);
        return 1;
    }

    ggml_backend_load_all();

    llama_model * model = llama_model_load_from_file(model_path, llama_model_default_params());
    if (!model) {
        fprintf(stderr, "gen3_a: load model failed\n");
        return 1;
    }

    const int32_t n_embd  = llama_model_n_embd(model);
    const int32_t n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model));

    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx   = 512;
    cparams.n_batch = 512;
    cparams.no_perf = true;
    llama_context * ctx = llama_init_from_model(model, cparams);
    if (!ctx) {
        fprintf(stderr, "gen3_a: create context failed\n");
        return 1;
    }

    llama_set_layer_range(ctx, 0, layer_end);

    const int b_fd = split_tcp_connect_retry(b_host, b_port, 300, 100);
    if (b_fd < 0) {
        fprintf(stderr, "gen3_a: connect to B failed %s:%d\n", b_host, b_port);
        return 1;
    }

    const int listen_fd = split_tcp_listen_host(bind_host, ctrl_port);
    if (listen_fd < 0) {
        fprintf(stderr, "gen3_a: ctrl listen failed port=%d\n", ctrl_port);
        return 1;
    }

    fprintf(stderr, "gen3_a: ready ctrl_port=%d layer_end=%d\n", ctrl_port, layer_end);

    const int ctrl_fd = split_tcp_accept(listen_fd);
#if !defined(_WIN32)
    close(listen_fd);
#endif
    if (ctrl_fd < 0) {
        fprintf(stderr, "gen3_a: ctrl accept failed\n");
        return 1;
    }

    while (true) {
        split_gen_a_req req{};
        std::vector<int32_t> tokens_i32;
        if (!split_gen_recv_req(ctrl_fd, req, tokens_i32)) {
            fprintf(stderr, "gen3_a: recv req failed\n");
            break;
        }

        if (req.cmd == SPLIT_GEN_CMD_SHUTDOWN) {
            split_ab_send_shutdown(b_fd);
            split_gen3_a_resp resp{};
            resp.magic   = SPLIT_GEN_MAGIC;
            resp.version = SPLIT_GEN3_VERSION;
            split_gen3_send_a_resp(ctrl_fd, resp, nullptr, 0);
            break;
        }

        if (req.cmd == SPLIT_GEN_CMD_RESET) {
            llama_memory_clear(llama_get_memory(ctx), true);
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

        if (req.cmd == SPLIT_GEN_CMD_PREFILL) {
            std::vector<llama_token> tokens((size_t) req.n_tokens);
            for (int32_t i = 0; i < req.n_tokens; ++i) {
                tokens[i] = (llama_token) tokens_i32[i];
            }
            decode_rc = split_gen_decode_tokens(ctx, tokens, req.pos_start, true);
        } else if (req.cmd == SPLIT_GEN_CMD_DECODE) {
            decode_rc = split_gen_decode_one(ctx, (llama_token) tokens_i32[0], req.pos_start);
        } else {
            fprintf(stderr, "gen3_a: unknown cmd %u\n", req.cmd);
            break;
        }

        const double ms_a = (ggml_time_us() - t0) / 1000.0;

        if (decode_rc != 0) {
            fprintf(stderr, "gen3_a: decode failed cmd=%u\n", req.cmd);
            break;
        }

        const int32_t n_out = (req.cmd == SPLIT_GEN_CMD_PREFILL) ? req.n_tokens : 1;
        const int32_t le    = req.layer_end > 0 ? req.layer_end : layer_end;

        split_gen3_a_resp resp{};
        std::vector<float> logits;
        if (!forward_to_b(b_fd, ctx, n_out, n_embd, le, req.pos_start,
                req.include_logits != 0, n_vocab, ms_a, resp, logits)) {
            break;
        }

        if (!split_gen3_send_a_resp(ctrl_fd, resp, logits.empty() ? nullptr : logits.data(), n_vocab)) {
            fprintf(stderr, "gen3_a: send resp failed\n");
            break;
        }
    }

#if !defined(_WIN32)
    close(ctrl_fd);
    close(b_fd);
#endif

    llama_free(ctx);
    llama_model_free(model);
    return 0;
}
