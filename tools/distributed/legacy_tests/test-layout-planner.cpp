#include "orchestrator/layout_planner/layout_planner.h"
#include "test_layout_common.h"

#include <cstdio>
#include <string>

int main() {
    const model_manifest manifest = make_test_manifest(16, 50 * 1024 * 1024);
    const std::vector<layout_node_input> nodes = {
        make_layout_node("gpu-a", 500.0, 8.0, 12.0, "cuda"),
        make_layout_node("gpu-b", 200.0, 8.0, 8.0, "metal"),
        make_layout_node("cpu-c", 50.0, 16.0, 0.0),
    };

    const auto result = build_desired_layout("test-model", manifest, nodes);
    if (!result.success) {
        fprintf(stderr, "test-layout-planner: build failed: %s\n", result.error.c_str());
        return 1;
    }
    if (!result.layout.fits_cluster) {
        fprintf(stderr, "test-layout-planner: expected fits_cluster=true\n");
        return 1;
    }
    if (static_cast<int32_t>(result.layout.placements.size()) != 16) {
        fprintf(stderr, "test-layout-planner: expected 16 placements\n");
        return 1;
    }

    std::string err;
    if (!validate_desired_layout(result.layout, manifest, err)) {
        fprintf(stderr, "test-layout-planner: validation failed: %s\n", err.c_str());
        return 1;
    }

    printf("test-layout-planner: OK placements=%zu\n", result.layout.placements.size());
    return 0;
}
