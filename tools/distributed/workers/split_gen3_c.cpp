// Process C: layers [layer_start, n_layer) - final stage, samples token

#include "ggml-backend.h"
#include "ggml.h"
#include "llama.h"
#include "split_gen_common.h"
#include "../transport/split_tcp_wire.h"

#include <cstdio>
#include <cstring>
#include <vector>

#if !defined(_WIN32)
#include <unistd.h>
#endif

static void usage(const char * prog) {
    fprintf(stderr, "usage: %s MODEL --bc-port PORT [--layer-start N]\n", prog);
}

int main(int argc, char ** argv) {
    if (argc < 2) {
        usage(argv[0]);
        return 1;
    }

    const char * model_path = argv[1];
    int bc_port     = -1;
    int layer_start = 8;
    const char * bind_host = "0.0.0.0";

    for (int i = 2; i < argc; ++i) {
        if (strcmp(argv[i], "--bc-port") == 0 && i + 1 < argc) {
            bc_port = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--bind") == 0 && i + 1 < argc) {
            bind_host = argv[++i];
        } else if (strcmp(argv[i], "--layer-start") == 0 && i + 1 < argc) {
            layer_start = atoi(argv[++i]);
        } else {
            usage(argv[0]);
            return 1;
        }
    }

    if (bc_port < 0) {
        usage(argv[0]);
        return 1;
    }

    const int listen_fd = split_tcp_listen_host(bind_host, bc_port);
    if (listen_fd < 0) {
        fprintf(stderr, "gen3_c: listen failed %s:%d\n", bind_host, bc_port);
        return 1;
    }

    ggml_backend_load_all();

    llama_model * model = llama_model_load_from_file(model_path, llama_model_default_params());
    if (!model) {
        fprintf(stderr, "gen3_c: load model failed\n");
        return 1;
    }

    const int32_t n_layer = llama_model_n_layer(model);
    const int32_t n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model));

    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx   = 512;
    cparams.n_batch = 512;
    cparams.no_perf = true;
    llama_context * ctx = llama_init_from_model(model, cparams);
    if (!ctx) {
        fprintf(stderr, "gen3_c: create context failed\n");
        return 1;
    }

    llama_set_layer_range(ctx, layer_start, n_layer);
    llama_sampler * smpl = split_gen_make_sampler();

    fprintf(stderr, "gen3_c: waiting bc_port=%d layer_start=%d\n", bc_port, layer_start);

    const int bc_fd = split_tcp_accept(listen_fd);
#if !defined(_WIN32)
    close(listen_fd);
#endif
    if (bc_fd < 0) {
        fprintf(stderr, "gen3_c: accept failed\n");
        return 1;
    }

    while (true) {
        split_ab_cmd cmd;
        if (!split_ab_recv_cmd(bc_fd, cmd)) {
            fprintf(stderr, "gen3_c: recv cmd failed\n");
            break;
        }

        if (cmd == SPLIT_AB_CMD_SHUTDOWN) {
            break;
        }

        if (cmd == SPLIT_AB_CMD_RESET) {
            llama_memory_clear(llama_get_memory(ctx), true);
            llama_clear_hidden_state(ctx);
            continue;
        }

        if (cmd != SPLIT_AB_CMD_HIDDEN) {
            fprintf(stderr, "gen3_c: unexpected cmd %u\n", cmd);
            break;
        }

        split_tcp_hidden_msg msg;
        if (!split_ab_recv_hidden(bc_fd, msg)) {
            fprintf(stderr, "gen3_c: recv hidden failed\n");
            break;
        }

        llama_set_layer_range(ctx, layer_start, n_layer);
        llama_set_hidden_state(ctx, msg.data.data(), msg.header.n_tokens);

        const int64_t t0 = ggml_time_us();

        if (split_gen_decode_hidden(ctx, msg.data.data(), msg.header.n_tokens, msg.header.n_embd,
                msg.meta.pos_start, true) != 0) {
            fprintf(stderr, "gen3_c: decode failed\n");
            break;
        }

        const double ms_compute = (ggml_time_us() - t0) / 1000.0;

        const int64_t t1 = ggml_time_us();
        const llama_token token_id = llama_sampler_sample(smpl, ctx, -1);
        llama_sampler_accept(smpl, token_id);
        const double ms_sample = (ggml_time_us() - t1) / 1000.0;

        split_gen3_c_resp resp{};
        resp.magic      = SPLIT_GEN_MAGIC;
        resp.token_id   = (int32_t) token_id;
        resp.n_vocab    = n_vocab;
        resp.ms_compute = ms_compute;
        resp.ms_sample  = ms_sample;

        if (!split_gen3_send_c_resp(bc_fd, resp)) {
            fprintf(stderr, "gen3_c: send resp failed\n");
            break;
        }
    }

#if !defined(_WIN32)
    close(bc_fd);
#endif

    llama_sampler_free(smpl);
    llama_free(ctx);
    llama_model_free(model);
    return 0;
}
