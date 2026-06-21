// Per-layer decode/prefill cost benchmark for scheduling data

#include "ggml-backend.h"
#include "ggml.h"
#include "llama.h"
#include "split_gen_common.h"

#include "../src/llama-ext.h"

#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

struct layer_stats {
    int    layer       = 0;
    double prefill_ms  = 0.0;
    double decode_ms   = 0.0;
    size_t memory_bytes = 0;
};

static size_t memory_total(const llama_context * ctx) {
    size_t total = 0;
    const llama_memory_breakdown mb = llama_get_memory_breakdown(ctx);
    for (const auto & kv : mb) {
        total += kv.second.total();
    }
    return total;
}

static double bench_prefill_layer(
        llama_context * ctx,
        const std::vector<llama_token> & prompt,
        int32_t n_embd,
        int layer) {
    llama_memory_clear(llama_get_memory(ctx), true);
    llama_clear_hidden_state(ctx);

    const int32_t n_tokens = (int32_t) prompt.size();
    std::vector<float> hidden_buf;

    if (layer > 0) {
        llama_set_layer_range(ctx, 0, layer);
        if (split_gen_decode_tokens(ctx, prompt, 0, true) != 0) {
            return -1.0;
        }
        const float * hidden = llama_get_embeddings(ctx);
        if (hidden == nullptr) {
            return -1.0;
        }
        hidden_buf.assign(hidden, hidden + (size_t) n_tokens * n_embd);
        llama_memory_clear(llama_get_memory(ctx), true);
    }

    const int64_t t0 = ggml_time_us();

    if (layer == 0) {
        llama_set_layer_range(ctx, 0, 1);
        if (split_gen_decode_tokens(ctx, prompt, 0, true) != 0) {
            return -1.0;
        }
    } else {
        llama_set_layer_range(ctx, layer, layer + 1);
        llama_set_hidden_state(ctx, hidden_buf.data(), n_tokens);
        if (split_gen_decode_hidden(ctx, hidden_buf.data(), n_tokens, n_embd, 0, true) != 0) {
            return -1.0;
        }
    }

    return (ggml_time_us() - t0) / 1000.0;
}

static double bench_decode_layer(
        llama_context * ctx,
        llama_token tok,
        llama_pos pos,
        int32_t n_embd,
        int layer,
        int n_iters) {
    double sum = 0.0;
    for (int i = 0; i < n_iters; ++i) {
        llama_memory_clear(llama_get_memory(ctx), true);
        llama_clear_hidden_state(ctx);

        std::vector<float> hidden_buf;
        if (layer > 0) {
            llama_set_layer_range(ctx, 0, layer);
            if (split_gen_decode_one(ctx, tok, pos) != 0) {
                return -1.0;
            }
            const float * hidden = llama_get_embeddings(ctx);
            if (hidden == nullptr) {
                return -1.0;
            }
            hidden_buf.assign(hidden, hidden + (size_t) n_embd);
            llama_memory_clear(llama_get_memory(ctx), true);
        }

        const int64_t t0 = ggml_time_us();

        if (layer == 0) {
            llama_set_layer_range(ctx, 0, 1);
            if (split_gen_decode_one(ctx, tok, pos) != 0) {
                return -1.0;
            }
        } else {
            llama_set_layer_range(ctx, layer, layer + 1);
            llama_set_hidden_state(ctx, hidden_buf.data(), 1);
            if (split_gen_decode_hidden(ctx, hidden_buf.data(), 1, n_embd, pos, true) != 0) {
                return -1.0;
            }
        }

        sum += (ggml_time_us() - t0) / 1000.0;
    }
    return sum / (double) n_iters;
}

static bool write_csv(const char * path, const std::vector<layer_stats> & stats) {
    FILE * f = fopen(path, "w");
    if (!f) {
        return false;
    }
    fprintf(f, "layer,prefill_ms,decode_ms,memory_bytes\n");
    for (const auto & s : stats) {
        fprintf(f, "%d,%.6f,%.6f,%zu\n", s.layer, s.prefill_ms, s.decode_ms, s.memory_bytes);
    }
    fclose(f);
    return true;
}

static void print_table(const std::vector<layer_stats> & stats) {
    printf("\nLayer  Prefill(ms)  Decode(ms)  Memory(KiB)\n");
    printf("-----  -----------  ----------  ----------\n");
    for (const auto & s : stats) {
        printf("%5d  %11.3f  %10.3f  %10.1f\n",
                s.layer, s.prefill_ms, s.decode_ms, s.memory_bytes / 1024.0);
    }
}

int main(int argc, char ** argv) {
    const char * model_path = nullptr;
    const char * csv_path   = "layer_cost.csv";
    int decode_iters = 10;

    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "--csv") == 0 && i + 1 < argc) {
            csv_path = argv[++i];
        } else if (strcmp(argv[i], "--decode-iters") == 0 && i + 1 < argc) {
            decode_iters = atoi(argv[++i]);
        } else if (model_path == nullptr) {
            model_path = argv[i];
        } else {
            fprintf(stderr, "usage: %s MODEL.gguf [--csv path] [--decode-iters N]\n", argv[0]);
            return 1;
        }
    }

    if (model_path == nullptr) {
        fprintf(stderr, "usage: %s MODEL.gguf [--csv path] [--decode-iters N]\n", argv[0]);
        return 1;
    }

    ggml_backend_load_all();

    llama_model * model = llama_model_load_from_file(model_path, llama_model_default_params());
    if (!model) {
        fprintf(stderr, "benchmark-layer-cost: load model failed\n");
        return 1;
    }

    const int32_t n_layer = llama_model_n_layer(model);
    const int32_t n_embd  = llama_model_n_embd(model);
    const llama_vocab * vocab = llama_model_get_vocab(model);
    const auto prompt = split_gen_tokenize(vocab, SPLIT_GEN_PROMPT);
    if (prompt.empty()) {
        fprintf(stderr, "benchmark-layer-cost: tokenize failed\n");
        return 1;
    }

    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx   = (uint32_t) std::max(prompt.size() + 8, (size_t) 512);
    cparams.n_batch = (uint32_t) std::max(prompt.size(), (size_t) 512);
    cparams.no_perf = true;

    llama_context * ctx = llama_init_from_model(model, cparams);
    if (!ctx) {
        fprintf(stderr, "benchmark-layer-cost: create context failed\n");
        return 1;
    }

    const llama_token decode_tok = prompt.back();
    const llama_pos decode_pos   = (llama_pos) (prompt.size() - 1);

    printf("benchmark-layer-cost: model=%s n_layer=%d n_prompt=%zu decode_iters=%d\n",
            model_path, n_layer, prompt.size(), decode_iters);

    std::vector<layer_stats> stats((size_t) n_layer);

    for (int32_t layer = 0; layer < n_layer; ++layer) {
        layer_stats s;
        s.layer = layer;

        s.prefill_ms = bench_prefill_layer(ctx, prompt, n_embd, layer);
        if (s.prefill_ms < 0) {
            fprintf(stderr, "benchmark-layer-cost: prefill failed layer=%d\n", layer);
            return 1;
        }

        s.decode_ms = bench_decode_layer(ctx, decode_tok, decode_pos, n_embd, layer, decode_iters);
        if (s.decode_ms < 0) {
            fprintf(stderr, "benchmark-layer-cost: decode failed layer=%d\n", layer);
            return 1;
        }

        s.memory_bytes = memory_total(ctx);
        stats[(size_t) layer] = s;

        printf("layer %2d: prefill=%.3f ms decode=%.3f ms mem=%.1f KiB\n",
                layer, s.prefill_ms, s.decode_ms, s.memory_bytes / 1024.0);
    }

    print_table(stats);

    if (!write_csv(csv_path, stats)) {
        fprintf(stderr, "benchmark-layer-cost: failed to write %s\n", csv_path);
        return 1;
    }

    printf("\nbenchmark-layer-cost: wrote %s\n", csv_path);

    llama_free(ctx);
    llama_model_free(model);
    return 0;
}
