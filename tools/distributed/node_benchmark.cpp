#include "node_benchmark.h"

#include "ggml-backend.h"
#include "ggml.h"
#include "llama.h"
#include "nlohmann/json.hpp"

#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <random>
#include <string>
#include <thread>
#include <vector>

#if !defined(_WIN32)
#include <sys/stat.h>
#include <unistd.h>
#endif

using json = nlohmann::json;

float dist_compute_benchmark_score(const float decode_tps, const float prefill_tps) {
    return decode_tps * 0.7f + prefill_tps * 0.3f;
}

std::string dist_benchmark_cache_path() {
    const char * home = std::getenv("HOME");
    if (home == nullptr || home[0] == '\0') {
        return ".distributed-llm/benchmark.json";
    }
    return std::string(home) + "/.distributed-llm/benchmark.json";
}

std::string dist_compute_model_hash(const std::string & model_path) {
    std::ifstream f(model_path, std::ios::binary);
    if (!f) {
        return {};
    }

    uint64_t hash = 14695981039346656037ULL;
    constexpr uint64_t prime = 1099511628211ULL;

    std::vector<char> buf(1024 * 1024);
    while (f) {
        f.read(buf.data(), (std::streamsize) buf.size());
        const std::streamsize n = f.gcount();
        for (std::streamsize i = 0; i < n; ++i) {
            hash ^= static_cast<unsigned char>(buf[(size_t) i]);
            hash *= prime;
        }
    }

#if !defined(_WIN32)
    struct stat st{};
    if (stat(model_path.c_str(), &st) == 0) {
        hash ^= static_cast<uint64_t>(st.st_size);
        hash *= prime;
        hash ^= static_cast<uint64_t>(st.st_mtime);
        hash *= prime;
    }
#endif

    char out[32];
    snprintf(out, sizeof(out), "%016llx", (unsigned long long) hash);
    return out;
}

static bool ensure_cache_dir(const std::string & path) {
    const auto slash = path.rfind('/');
    if (slash == std::string::npos) {
        return true;
    }
    const std::string dir = path.substr(0, slash);
#if !defined(_WIN32)
    if (mkdir(dir.c_str(), 0755) == 0 || errno == EEXIST) {
        return true;
    }
#endif
    return false;
}

bool dist_load_benchmark_cache(const std::string & model_path, BenchmarkResult & out) {
    std::ifstream f(dist_benchmark_cache_path());
    if (!f) {
        return false;
    }

    try {
        json j;
        f >> j;

        const std::string model_hash = dist_compute_model_hash(model_path);
        if (j.value("model_hash", "") != model_hash) {
            return false;
        }
        if (j.value("benchmark_version", "") != DIST_BENCHMARK_VERSION) {
            return false;
        }

        out.decode_tps  = j.value("decode_tps", 0.0f);
        out.prefill_tps = j.value("prefill_tps", 0.0f);
        out.load_ms     = j.value("load_ms", 0.0f);
        out.score       = j.value("score", 0.0f);
        out.n_layer     = j.value("n_layer", 0);
        out.n_embd      = j.value("n_embd", 0);

        if (out.decode_tps <= 0.0f || out.prefill_tps <= 0.0f || out.score <= 0.0f) {
            return false;
        }
        if (out.n_layer <= 0 || out.n_embd <= 0) {
            return false;
        }
        return true;
    } catch (...) {
        return false;
    }
}

bool dist_save_benchmark_cache(const std::string & model_path, const BenchmarkResult & result) {
    const std::string path = dist_benchmark_cache_path();
    if (!ensure_cache_dir(path)) {
        return false;
    }

    json j = {
        { "model_hash", dist_compute_model_hash(model_path) },
        { "benchmark_version", DIST_BENCHMARK_VERSION },
        { "decode_tps", result.decode_tps },
        { "prefill_tps", result.prefill_tps },
        { "load_ms", result.load_ms },
        { "score", result.score },
        { "n_layer", result.n_layer },
        { "n_embd", result.n_embd },
    };

    std::ofstream f(path);
    if (!f) {
        return false;
    }
    f << j.dump(2);
    return true;
}

static bool bench_prefill_tokens(
        llama_context * ctx,
        const int n_prompt,
        const int n_batch,
        const int n_threads,
        float & out_tps) {
    llama_set_n_threads(ctx, n_threads, n_threads);

    const llama_model * model   = llama_get_model(ctx);
    const llama_vocab * vocab   = llama_model_get_vocab(model);
    const int32_t       n_vocab = llama_vocab_n_tokens(vocab);

    std::vector<llama_token> tokens((size_t) n_batch);
    std::mt19937 rng(1234);

    llama_memory_clear(llama_get_memory(ctx), true);

    int n_processed = 0;
    const int64_t t0 = ggml_time_us();

    while (n_processed < n_prompt) {
        const int n_tokens = std::min(n_prompt - n_processed, n_batch);
        tokens[0] = (n_processed == 0 && llama_vocab_get_add_bos(vocab))
                ? llama_vocab_bos(vocab)
                : (llama_token) (rng() % n_vocab);
        for (int i = 1; i < n_tokens; ++i) {
            tokens[i] = (llama_token) (rng() % n_vocab);
        }
        if (llama_decode(ctx, llama_batch_get_one(tokens.data(), n_tokens)) != 0) {
            return false;
        }
        n_processed += n_tokens;
    }

    llama_synchronize(ctx);
    const double elapsed_s = (ggml_time_us() - t0) / 1e6;
    if (elapsed_s <= 0.0) {
        return false;
    }

    out_tps = (float) (n_prompt / elapsed_s);
    return true;
}

static bool bench_decode_tokens(
        llama_context * ctx,
        const int n_prompt,
        const int n_gen,
        const int n_batch,
        const int n_threads,
        float & out_tps) {
    llama_set_n_threads(ctx, n_threads, n_threads);

    const llama_model * model   = llama_get_model(ctx);
    const llama_vocab * vocab   = llama_model_get_vocab(model);
    const int32_t       n_vocab = llama_vocab_n_tokens(vocab);

    std::vector<llama_token> tokens((size_t) n_batch);
    std::mt19937 rng(5678);

    llama_memory_clear(llama_get_memory(ctx), true);

    int n_processed = 0;
    while (n_processed < n_prompt) {
        const int n_tokens = std::min(n_prompt - n_processed, n_batch);
        tokens[0] = (n_processed == 0 && llama_vocab_get_add_bos(vocab))
                ? llama_vocab_bos(vocab)
                : (llama_token) (rng() % n_vocab);
        for (int i = 1; i < n_tokens; ++i) {
            tokens[i] = (llama_token) (rng() % n_vocab);
        }
        if (llama_decode(ctx, llama_batch_get_one(tokens.data(), n_tokens)) != 0) {
            return false;
        }
        n_processed += n_tokens;
    }

    llama_token token = (llama_token) (rng() % n_vocab);
    const int64_t t0 = ggml_time_us();

    for (int i = 0; i < n_gen; ++i) {
        if (llama_decode(ctx, llama_batch_get_one(&token, 1)) != 0) {
            return false;
        }
        llama_synchronize(ctx);
        token = (llama_token) (rng() % n_vocab);
    }

    const double elapsed_s = (ggml_time_us() - t0) / 1e6;
    if (elapsed_s <= 0.0) {
        return false;
    }

    out_tps = (float) (n_gen / elapsed_s);
    return true;
}

BenchmarkResult run_node_benchmark(const std::string & model_path) {
    BenchmarkResult result{};

    ggml_backend_load_all();

    const int64_t t_load0 = ggml_time_us();
    llama_model * model = llama_model_load_from_file(model_path.c_str(), llama_model_default_params());
    if (!model) {
        fprintf(stderr, "node_benchmark: failed to load model %s\n", model_path.c_str());
        return result;
    }
    result.n_layer = llama_model_n_layer(model);
    result.n_embd  = llama_model_n_embd(model);
    result.load_ms = (float) ((ggml_time_us() - t_load0) / 1000.0);

    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx    = std::max(2048, DIST_BENCH_PREFILL_TOKENS + 64);
    cparams.n_batch  = std::min(512, DIST_BENCH_PREFILL_TOKENS);
    cparams.n_ubatch = cparams.n_batch;
    cparams.no_perf  = true;

    llama_context * ctx = llama_init_from_model(model, cparams);
    if (!ctx) {
        fprintf(stderr, "node_benchmark: failed to create context\n");
        llama_model_free(model);
        return BenchmarkResult{};
    }

    int n_threads = (int) std::thread::hardware_concurrency();
    if (n_threads <= 0) {
        n_threads = 4;
    }

    if (!bench_prefill_tokens(ctx, DIST_BENCH_PREFILL_TOKENS, cparams.n_batch, n_threads, result.prefill_tps)) {
        fprintf(stderr, "node_benchmark: prefill benchmark failed\n");
        llama_free(ctx);
        llama_model_free(model);
        return BenchmarkResult{};
    }

    if (!bench_decode_tokens(
                ctx,
                DIST_BENCH_DECODE_PROMPT_TOKENS,
                DIST_BENCH_DECODE_GEN_TOKENS,
                cparams.n_batch,
                n_threads,
                result.decode_tps)) {
        fprintf(stderr, "node_benchmark: decode benchmark failed\n");
        llama_free(ctx);
        llama_model_free(model);
        return BenchmarkResult{};
    }

    result.score = dist_compute_benchmark_score(result.decode_tps, result.prefill_tps);

    fprintf(stderr,
            "node_benchmark: decode_tps=%.1f prefill_tps=%.1f load_ms=%.0f score=%.1f\n",
            result.decode_tps, result.prefill_tps, result.load_ms, result.score);

    llama_free(ctx);
    llama_model_free(model);
    return result;
}

BenchmarkResult dist_get_or_run_benchmark(const std::string & model_path, const bool rebenchmark) {
    BenchmarkResult cached{};
    if (!rebenchmark && dist_load_benchmark_cache(model_path, cached)) {
        fprintf(stderr, "node_benchmark: using cached score=%.1f (decode=%.1f prefill=%.1f)\n",
                cached.score, cached.decode_tps, cached.prefill_tps);
        return cached;
    }

    BenchmarkResult result = run_node_benchmark(model_path);
    if (result.score > 0.0f) {
        dist_save_benchmark_cache(model_path, result);
    }
    return result;
}
