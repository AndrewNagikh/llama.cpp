#include "memory_estimator.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>

static bool approx_eq(double a, double b, double eps = 0.01) {
    return std::fabs(a - b) <= eps;
}

static bool test_catalog_estimate() {
    model_info model{};
    model.id          = "qwen3-70b";
    model.display_name = "Qwen3 70B";
    model.n_layers    = 80;
    model.n_embd      = 8192;
    model.size_gb     = 42.0;

    const auto mem = estimate_model_memory_from_catalog(model, 4096);

    if (mem.n_layer != 80) {
        fprintf(stderr, "memory-estimator: expected n_layer=80 got %d\n", mem.n_layer);
        return false;
    }
    if (mem.layers.size() != 80) {
        fprintf(stderr, "memory-estimator: expected 80 layer descriptors got %zu\n", mem.layers.size());
        return false;
    }
    if (mem.weights_bytes == 0) {
        fprintf(stderr, "memory-estimator: weights_bytes is zero\n");
        return false;
    }
    if (mem.kv_bytes == 0) {
        fprintf(stderr, "memory-estimator: kv_bytes is zero\n");
        return false;
    }
    if (mem.total_bytes() != mem.weights_bytes + mem.kv_bytes + mem.compute_bytes + mem.scratch_bytes) {
        fprintf(stderr, "memory-estimator: total_bytes mismatch\n");
        return false;
    }

    // 42 GB weights / 80 layers ~= 0.525 GB per layer.
    const double layer_gb = dist_bytes_to_gb(mem.layers[0].weight_bytes);
    if (!approx_eq(layer_gb, 42.0 / 80.0, 0.05)) {
        fprintf(stderr, "memory-estimator: per-layer weight %.2f GB unexpected\n", layer_gb);
        return false;
    }

    if (!approx_eq(mem.weights_gb(), 42.0, 0.05)) {
        fprintf(stderr, "memory-estimator: weights_gb %.2f != 42.0\n", mem.weights_gb());
        return false;
    }

    return true;
}

static bool test_invalid_gguf() {
    const auto mem = estimate_model_memory("/nonexistent/path/model.gguf", 4096);
    if (mem.valid()) {
        fprintf(stderr, "memory-estimator: nonexistent GGUF should be invalid\n");
        return false;
    }
    return true;
}

static bool test_optional_gguf() {
    const char * env = std::getenv("MODEL");
    if (!env || env[0] == '\0') {
        return true;
    }
    const auto mem = estimate_model_memory(env, 4096);
    if (!mem.valid()) {
        fprintf(stderr, "memory-estimator: failed to estimate %s\n", env);
        return false;
    }
    printf("memory-estimator: %s weights=%.1f GB kv=%.1f GB required=%.1f GB\n",
           env, mem.weights_gb(), mem.kv_gb(), mem.total_gb());
    return true;
}

int main() {
    bool ok = true;
    ok &= test_catalog_estimate();
    ok &= test_invalid_gguf();
    ok &= test_optional_gguf();

    if (!ok) {
        fprintf(stderr, "test-memory-estimator: FAILED\n");
        return 1;
    }
    printf("test-memory-estimator: OK\n");
    return 0;
}
