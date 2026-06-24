// Process A: layers [0, layer_end) -> hidden state -> TCP

#include "ggml-backend.h"
#include "ggml.h"
#include "llama.h"
#include "split_tcp_wire.h"

#include "llama-distributed.h"

#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#if !defined(_WIN32)
#include <unistd.h>
#endif

static void usage(const char * prog) {
    fprintf(stderr,
            "usage: %s MODEL [--host HOST] [--port PORT] [--prompt TEXT] [--layer-end N]\n",
            prog);
}

static int decode_tokens(llama_context * ctx, const std::vector<llama_token> & tokens) {
    const int32_t n_tokens = (int32_t) tokens.size();
    llama_batch batch = llama_batch_init(n_tokens, 0, 1);
    for (int32_t i = 0; i < n_tokens; ++i) {
        batch.token[i]     = tokens[i];
        batch.pos[i]       = i;
        batch.n_seq_id[i]  = 1;
        batch.seq_id[i][0] = 0;
        batch.logits[i]    = 1;
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
    const char * host       = "127.0.0.1";
    int          port       = 18765;
    std::string  prompt     = "Hello";
    int          layer_end  = -1;

    for (int i = 2; i < argc; ++i) {
        if (strcmp(argv[i], "--host") == 0 && i + 1 < argc) {
            host = argv[++i];
        } else if (strcmp(argv[i], "--port") == 0 && i + 1 < argc) {
            port = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--prompt") == 0 && i + 1 < argc) {
            prompt = argv[++i];
        } else if (strcmp(argv[i], "--layer-end") == 0 && i + 1 < argc) {
            layer_end = atoi(argv[++i]);
        } else {
            usage(argv[0]);
            return 1;
        }
    }

    ggml_backend_load_all();

    llama_model * model = llama_model_load_from_file(model_path, llama_model_default_params());
    if (!model) {
        fprintf(stderr, "failed to load model\n");
        return 1;
    }

    const int32_t n_layer = llama_model_n_layer(model);
    if (layer_end < 0) {
        layer_end = n_layer / 2;
    }

    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx    = 512;
    cparams.no_perf  = true;
    llama_context * ctx = llama_init_from_model(model, cparams);
    if (!ctx) {
        fprintf(stderr, "failed to create context\n");
        return 1;
    }

    const llama_vocab * vocab = llama_model_get_vocab(model);
    const int n_tokens = -llama_tokenize(vocab, prompt.c_str(), prompt.size(), nullptr, 0, true, true);
    std::vector<llama_token> tokens(n_tokens);
    llama_tokenize(vocab, prompt.c_str(), prompt.size(), tokens.data(), tokens.size(), true, true);

    const int32_t n_embd = llama_model_n_embd(model);

    llama_set_layer_range(ctx, 0, layer_end);
    llama_memory_clear(llama_get_memory(ctx), true);

    const int64_t t0 = ggml_time_us();
    if (decode_tokens(ctx, tokens) != 0) {
        fprintf(stderr, "sender: decode failed\n");
        return 1;
    }
    const int64_t t1 = ggml_time_us();

    const float * hidden = llama_get_embeddings(ctx);
    if (hidden == nullptr) {
        fprintf(stderr, "sender: no hidden state\n");
        return 1;
    }

    const int conn = split_tcp_connect(host, port);
    if (conn < 0) {
        fprintf(stderr, "sender: TCP connect failed (host=%s port=%d)\n", host, port);
        return 1;
    }

    const int64_t t2 = ggml_time_us();
    if (!split_tcp_send_hidden(conn, n_tokens, n_embd, layer_end, hidden)) {
        fprintf(stderr, "sender: send failed\n");
        return 1;
    }
    const int64_t t3 = ggml_time_us();

#if !defined(_WIN32)
    close(conn);
#endif

    split_tcp_perf perf{};
    perf.ms_compute = (t1 - t0) / 1000.0;
    perf.ms_send    = (t3 - t2) / 1000.0;
    perf.ms_total   = (t3 - t0) / 1000.0;
    perf.nbytes     = sizeof(split_tcp_header) + (size_t) n_tokens * (size_t) n_embd * sizeof(float);

    printf("split_sender: layer_end=%d n_tokens=%d n_embd=%d\n", layer_end, n_tokens, n_embd);
    printf("perf_sender: compute_ms=%.3f send_ms=%.3f total_ms=%.3f nbytes=%zu\n",
            perf.ms_compute, perf.ms_send, perf.ms_total, perf.nbytes);

    llama_free(ctx);
    llama_model_free(model);
    return 0;
}
