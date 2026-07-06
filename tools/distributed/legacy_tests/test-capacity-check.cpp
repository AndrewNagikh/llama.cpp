#include "memory_estimator.h"

#include <cstdio>
#include <string>
#include <vector>

static model_memory_requirements make_model(
        int32_t n_layer,
        double  weights_gb,
        double  kv_gb,
        double  compute_gb,
        double  scratch_gb) {
    model_memory_requirements mem{};
    mem.n_layer  = n_layer;
    mem.weights_bytes  = static_cast<uint64_t>(weights_gb  * 1024 * 1024 * 1024);
    mem.kv_bytes       = static_cast<uint64_t>(kv_gb       * 1024 * 1024 * 1024);
    mem.compute_bytes  = static_cast<uint64_t>(compute_gb  * 1024 * 1024 * 1024);
    mem.scratch_bytes  = static_cast<uint64_t>(scratch_gb  * 1024 * 1024 * 1024);
    return mem;
}

static dist_node_info make_node(const std::string & id, double ram_gb, double vram_gb) {
    dist_node_info n{};
    n.node_id = id;
    n.online  = true;
    n.memory.free_ram_bytes  = static_cast<uint64_t>(ram_gb  * 1024 * 1024 * 1024);
    n.memory.free_vram_bytes = static_cast<uint64_t>(vram_gb * 1024 * 1024 * 1024);
    n.memory.has_gpu = (vram_gb > 0.0);
    return n;
}

int main() {
    bool ok = true;

    const auto model = make_model(80, 40.0, 12.0, 4.0, 2.0);
    const double required = model.total_gb();
    printf("test-capacity-check: required=%.1f GB\n", required);

    // Should fit: 60 GB model in a 16 GB RAM CPU + 56 GB GPU cluster.
    {
        std::vector<dist_node_info> nodes = {
            make_node("cpu-a", 16.0, 0.0),
            make_node("gpu-b", 0.0, 56.0),
        };
        const auto fit = dist_check_cluster_memory_fit(model, nodes);
        printf("  fit: required=%.1f available=%.1f missing=%.1f fits=%d\n",
               fit.required_gb, fit.available_gb, fit.missing_gb, fit.fits ? 1 : 0);
        if (!fit.fits) {
            fprintf(stderr, "test-capacity-check: expected fit=true\n");
            ok = false;
        }
        if (fit.required_gb != required) {
            fprintf(stderr, "test-capacity-check: required mismatch\n");
            ok = false;
        }
        if (fit.available_gb < 69.0 || fit.available_gb > 73.0) {
            fprintf(stderr, "test-capacity-check: available_gb unexpected %.1f\n", fit.available_gb);
            ok = false;
        }
    }

    // Should not fit: two 8 GB CPU nodes give only 16 GB primary budget.
    {
        std::vector<dist_node_info> nodes = {
            make_node("cpu-a", 8.0, 0.0),
            make_node("cpu-b", 8.0, 0.0),
        };
        const auto fit = dist_check_cluster_memory_fit(model, nodes);
        printf("  fit: required=%.1f available=%.1f missing=%.1f fits=%d\n",
               fit.required_gb, fit.available_gb, fit.missing_gb, fit.fits ? 1 : 0);
        if (fit.fits) {
            fprintf(stderr, "test-capacity-check: expected fit=false\n");
            ok = false;
        }
        if (fit.missing_gb <= 0.0) {
            fprintf(stderr, "test-capacity-check: expected missing_gb > 0\n");
            ok = false;
        }
    }

    if (!ok) {
        fprintf(stderr, "test-capacity-check: FAILED\n");
        return 1;
    }
    printf("test-capacity-check: OK\n");
    return 0;
}
