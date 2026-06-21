// Process B: TCP -> hidden state -> layers [layer_end, n_layer) -> logits

#include "ggml-backend.h"
#include "ggml.h"
#include "llama.h"
#include "split_tcp_wire.h"

#include "../src/llama-ext.h"

#include <cstdio>
#include <cstring>
#include <vector>

#if !defined(_WIN32)
#include <unistd.h>
#endif

static void usage(const char * prog) {
    fprintf(stderr, "usage: %s MODEL [--port PORT] [--out FILE]\n", prog);
}

static int decode_hidden(llama_context * ctx, const float * hidden, int32_t n_tokens, int32_t n_embd) {
    llama_batch batch = llama_batch_init(n_tokens, n_embd, 1);
    for (int32_t i = 0; i < n_tokens; ++i) {
        batch.pos[i]       = i;
        batch.n_seq_id[i]  = 1;
        batch.seq_id[i][0] = 0;
        batch.logits[i]    = (i == n_tokens - 1);
        std::memcpy(
                batch.embd + (size_t) i * n_embd,
                hidden + (size_t) i * n_embd,
                (size_t) n_embd * sizeof(float));
    }
    batch.n_tokens = n_tokens;
    const int ret = llama_decode(ctx, batch);
    llama_batch_free(batch);
    return ret;
}

int main(int argc, char ** argv) {
    if (argc < 2) {
        usage(argv[0]);
        return 1;
    }

    const char * model_path = argv[1];
    int          port       = 18765;
    const char * out_path   = nullptr;

    for (int i = 2; i < argc; ++i) {
        if (strcmp(argv[i], "--port") == 0 && i + 1 < argc) {
            port = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--out") == 0 && i + 1 < argc) {
            out_path = argv[++i];
        } else {
            usage(argv[0]);
            return 1;
        }
    }

    const int listen_fd = split_tcp_listen(port);
    if (listen_fd < 0) {
        fprintf(stderr, "receiver: listen failed on port %d\n", port);
        return 1;
    }

    ggml_backend_load_all();

    llama_model * model = llama_model_load_from_file(model_path, llama_model_default_params());
    if (!model) {
        fprintf(stderr, "receiver: failed to load model\n");
        return 1;
    }

    const int32_t n_layer = llama_model_n_layer(model);

    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx   = 512;
    cparams.no_perf = true;
    llama_context * ctx = llama_init_from_model(model, cparams);
    if (!ctx) {
        fprintf(stderr, "receiver: failed to create context\n");
        return 1;
    }

    const int conn = split_tcp_accept(listen_fd);
#if !defined(_WIN32)
    close(listen_fd);
#endif
    if (conn < 0) {
        fprintf(stderr, "receiver: accept failed\n");
        return 1;
    }

    const int64_t t0 = ggml_time_us();

    split_tcp_hidden_msg msg;
    if (!split_tcp_recv_hidden(conn, msg)) {
        fprintf(stderr, "receiver: recv failed\n");
        return 1;
    }
    const int64_t t1 = ggml_time_us();

#if !defined(_WIN32)
    close(conn);
#endif

    const int32_t layer_end = msg.header.layer_end;

    llama_set_layer_range(ctx, layer_end, n_layer);
    llama_memory_clear(llama_get_memory(ctx), true);
    llama_set_hidden_state(ctx, msg.data.data(), msg.header.n_tokens);

    const int64_t t2 = ggml_time_us();
    if (decode_hidden(ctx, msg.data.data(), msg.header.n_tokens, msg.header.n_embd) != 0) {
        fprintf(stderr, "receiver: decode failed\n");
        return 1;
    }
    const int64_t t3 = ggml_time_us();

    const int32_t n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model));
    const float * logits = llama_get_logits_ith(ctx, -1);
    if (logits == nullptr) {
        fprintf(stderr, "receiver: no logits\n");
        return 1;
    }

    split_tcp_result_file meta{};
    meta.n_vocab      = n_vocab;
    meta.n_tokens     = msg.header.n_tokens;
    meta.n_embd       = msg.header.n_embd;
    meta.layer_start  = layer_end;
    meta.layer_end    = n_layer;
    meta.ms_recv      = (t1 - t0) / 1000.0;
    meta.ms_decode    = (t3 - t2) / 1000.0;
    meta.ms_total     = (t3 - t0) / 1000.0;

    if (out_path != nullptr) {
        if (!split_tcp_write_result(out_path, meta, logits)) {
            fprintf(stderr, "receiver: failed to write %s\n", out_path);
            return 1;
        }
    }

    printf("split_receiver: layer_start=%d layer_end=%d n_tokens=%d n_embd=%d\n",
            layer_end, n_layer, msg.header.n_tokens, msg.header.n_embd);
    printf("perf_receiver: recv_ms=%.3f decode_ms=%.3f total_ms=%.3f recv_bytes=%zu\n",
            meta.ms_recv, meta.ms_decode, meta.ms_total,
            sizeof(split_tcp_header) + msg.data.size() * sizeof(float));
    printf("logits[0..3]=%f %f %f %f\n", logits[0], logits[1], logits[2], logits[3]);

    llama_free(ctx);
    llama_model_free(model);
    return 0;
}
