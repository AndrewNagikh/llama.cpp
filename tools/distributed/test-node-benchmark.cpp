#include "node_benchmark.h"

#include <cstdio>
#include <cstring>

int main(int argc, char ** argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: %s MODEL.gguf [--rebenchmark]\n", argv[0]);
        return 1;
    }

    const char * model_path = argv[1];
    bool rebenchmark = false;
    for (int i = 2; i < argc; ++i) {
        if (strcmp(argv[i], "--rebenchmark") == 0) {
            rebenchmark = true;
        }
    }

    const BenchmarkResult result = dist_get_or_run_benchmark(model_path, rebenchmark);

    if (result.decode_tps <= 0.0f) {
        fprintf(stderr, "test-node-benchmark: decode_tps <= 0\n");
        return 1;
    }
    if (result.prefill_tps <= 0.0f) {
        fprintf(stderr, "test-node-benchmark: prefill_tps <= 0\n");
        return 1;
    }
    if (result.score <= 0.0f) {
        fprintf(stderr, "test-node-benchmark: score <= 0\n");
        return 1;
    }

    const float expected = dist_compute_benchmark_score(result.decode_tps, result.prefill_tps);
    if (result.score < expected - 0.01f || result.score > expected + 0.01f) {
        fprintf(stderr, "test-node-benchmark: score mismatch got=%.3f expected=%.3f\n",
                result.score, expected);
        return 1;
    }

    printf("test-node-benchmark: OK decode=%.1f prefill=%.1f load=%.0fms score=%.1f\n",
            result.decode_tps, result.prefill_tps, result.load_ms, result.score);
    return 0;
}
