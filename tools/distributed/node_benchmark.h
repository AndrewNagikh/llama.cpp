#pragma once

#include <cstdint>
#include <string>

// Bump when benchmark methodology changes (invalidates cache).
static constexpr const char * DIST_BENCHMARK_VERSION = "1.0";

static constexpr int DIST_BENCH_PREFILL_TOKENS      = 512;
static constexpr int DIST_BENCH_DECODE_PROMPT_TOKENS = 32;
static constexpr int DIST_BENCH_DECODE_GEN_TOKENS    = 64;

struct BenchmarkResult {
    float decode_tps  = 0.0f;
    float prefill_tps = 0.0f;
    float load_ms     = 0.0f;
    float score       = 0.0f;
    int32_t n_layer   = 0;
    int32_t n_embd    = 0;
};

float dist_compute_benchmark_score(float decode_tps, float prefill_tps);

// Run full benchmark (load + prefill + decode). Returns zeros on failure.
BenchmarkResult run_node_benchmark(const std::string & model_path);

// Load cache from ~/.distributed-llm/benchmark.json if valid.
bool dist_load_benchmark_cache(
        const std::string & model_path,
        BenchmarkResult & out);

// Save benchmark result to cache.
bool dist_save_benchmark_cache(
        const std::string & model_path,
        const BenchmarkResult & result);

// Use cache when model_hash and benchmark version match; rerun when rebenchmark=true.
BenchmarkResult dist_get_or_run_benchmark(
        const std::string & model_path,
        bool rebenchmark);

// Hardware-only score when no local GGUF is available (layer-first nodes).
BenchmarkResult dist_run_hardware_benchmark();

// Model benchmark when path is set; otherwise hardware-only estimate.
BenchmarkResult dist_get_or_run_benchmark_optional(
        const std::string & model_path,
        bool rebenchmark);

std::string dist_benchmark_cache_path();
std::string dist_compute_model_hash(const std::string & model_path);
