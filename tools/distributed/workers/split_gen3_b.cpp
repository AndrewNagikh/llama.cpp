// Process B: middle stage - layers [layer_start, layer_end)

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
    fprintf(stderr, "usage: %s MODEL --ab-port PORT --bc-port PORT [--layer-start N] [--layer-end N]\n", prog);
}

static bool forward_to_c(
        int bc_fd,
        int32_t n_tokens,
        int32_t n_embd,
        int32_t layer_end,
        int32_t pos_start,
        double ms_b,
        const float * hidden,
        split_gen3_mid_resp & resp) {
    if (hidden == nullptr && n_tokens > 0) {
        fprintf(stderr, "gen3_b: missing hidden state\n");
        return false;
    }

    const int64_t t0 = ggml_time_us();
    if (!split_ab_send_hidden(bc_fd, n_tokens, n_embd, layer_end, pos_start, 0, hidden)) {
        fprintf(stderr, "gen3_b: send hidden to C failed\n");
        return false;
    }
    const int64_t t1 = ggml_time_us();

    split_gen3_c_resp cresp{};
    if (!split_gen3_recv_c_resp(bc_fd, cresp)) {
        fprintf(stderr, "gen3_b: recv C resp failed\n");
        return false;
    }

    resp.magic       = SPLIT_GEN_MAGIC;
    resp.token_id    = cresp.token_id;
    resp.n_vocab     = cresp.n_vocab;
    resp.ms_b_compute = ms_b;
    resp.ms_bc_xfer   = (t1 - t0) / 1000.0;
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

int main(int argc, char ** argv) {
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
        } else {
            usage(argv[0]);
            return 1;
        }
    }

    if (ab_port < 0 || bc_port < 0) {
        usage(argv[0]);
        return 1;
    }

    const int listen_fd = split_tcp_listen_host(bind_host, ab_port);
    if (listen_fd < 0) {
        fprintf(stderr, "gen3_b: listen failed %s:%d\n", bind_host, ab_port);
        return 1;
    }

    ggml_backend_load_all();

    llama_model * model = llama_model_load_from_file(model_path, llama_model_default_params());
    if (!model) {
        fprintf(stderr, "gen3_b: load model failed\n");
        return 1;
    }

    const int32_t n_embd = llama_model_n_embd(model);

    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx   = 512;
    cparams.n_batch = 512;
    cparams.no_perf = true;
    llama_context * ctx = llama_init_from_model(model, cparams);
    if (!ctx) {
        fprintf(stderr, "gen3_b: create context failed\n");
        return 1;
    }

    llama_set_layer_range(ctx, layer_start, layer_end);

    const int bc_fd = split_tcp_connect_retry(bc_host, bc_port, 300, 100);
    if (bc_fd < 0) {
        fprintf(stderr, "gen3_b: connect to C failed %s:%d\n", bc_host, bc_port);
        return 1;
    }

    fprintf(stderr, "gen3_b: ready ab_port=%d layer=[%d,%d)\n", ab_port, layer_start, layer_end);

    const int ab_fd = split_tcp_accept(listen_fd);
#if !defined(_WIN32)
    close(listen_fd);
#endif
    if (ab_fd < 0) {
        fprintf(stderr, "gen3_b: accept failed\n");
        return 1;
    }

    while (true) {
        split_ab_cmd cmd;
        if (!split_ab_recv_cmd(ab_fd, cmd)) {
            fprintf(stderr, "gen3_b: recv cmd failed\n");
            break;
        }

        if (cmd == SPLIT_AB_CMD_SHUTDOWN) {
            split_ab_send_shutdown(bc_fd);
            break;
        }

        if (cmd == SPLIT_AB_CMD_RESET) {
            llama_memory_clear(llama_get_memory(ctx), true);
            llama_clear_hidden_state(ctx);
            split_ab_send_reset(bc_fd);
            continue;
        }

        if (cmd != SPLIT_AB_CMD_HIDDEN) {
            fprintf(stderr, "gen3_b: unexpected cmd %u\n", cmd);
            break;
        }

        split_tcp_hidden_msg msg;
        if (!split_ab_recv_hidden(ab_fd, msg)) {
            fprintf(stderr, "gen3_b: recv hidden failed\n");
            break;
        }

        llama_set_layer_range(ctx, layer_start, layer_end);

        double ms_b = 0.0;
        std::vector<float> out_hidden;
        if (!run_b_layers(ctx, msg, n_embd, layer_start, layer_end, ms_b, out_hidden)) {
            fprintf(stderr, "gen3_b: decode failed\n");
            break;
        }

        split_gen3_mid_resp resp{};
        if (!forward_to_c(bc_fd, msg.header.n_tokens, n_embd, layer_end,
                msg.meta.pos_start, ms_b, out_hidden.data(), resp)) {
            break;
        }

        if (!split_gen3_send_mid_resp(ab_fd, resp)) {
            fprintf(stderr, "gen3_b: send mid resp failed\n");
            break;
        }
    }

#if !defined(_WIN32)
    close(ab_fd);
    close(bc_fd);
#endif

    llama_free(ctx);
    llama_model_free(model);
    return 0;
}
