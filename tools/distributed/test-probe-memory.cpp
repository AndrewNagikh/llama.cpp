#include "dist_common.h"

#include <cstdio>

int main() {
    dist_node_memory mem{};
    dist_probe_node_memory(mem);

    if (mem.total_ram_bytes == 0) {
        fprintf(stderr, "test-probe-memory: total_ram_bytes is zero\n");
        return 1;
    }
    if (mem.free_ram_bytes == 0) {
        fprintf(stderr, "test-probe-memory: free_ram_bytes is zero\n");
        return 1;
    }
    if (mem.free_ram_bytes > mem.total_ram_bytes) {
        fprintf(stderr, "test-probe-memory: free_ram exceeds total_ram\n");
        return 1;
    }

#if defined(__APPLE__)
    // The old free_count-only formula often reported <1 GB on 16-18 GB Macs.
    const uint64_t min_expected = mem.total_ram_bytes / 10; // at least 10%
    if (mem.free_ram_bytes < min_expected) {
        fprintf(stderr, "test-probe-memory: available RAM %.2f GB looks too low for total %.2f GB\n",
                dist_bytes_to_gb(mem.free_ram_bytes),
                dist_bytes_to_gb(mem.total_ram_bytes));
        return 1;
    }
#endif

    printf("test-probe-memory: OK total_ram=%.2f GB available_ram=%.2f GB",
            dist_bytes_to_gb(mem.total_ram_bytes),
            dist_bytes_to_gb(mem.free_ram_bytes));
    if (mem.has_gpu) {
        printf(" total_vram=%.2f GB free_vram=%.2f GB",
                dist_bytes_to_gb(mem.total_vram_bytes),
                dist_bytes_to_gb(mem.free_vram_bytes));
    }
    printf("\n");
    return 0;
}
