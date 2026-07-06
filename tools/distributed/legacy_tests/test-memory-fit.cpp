#include "orchestrator/layout_planner/layout_planner.h"
#include "test_layout_common.h"

#include <cstdio>

int main() {
    const model_manifest manifest = make_test_manifest(80, 500 * 1024 * 1024);
    const std::vector<layout_node_input> nodes = {
        make_layout_node("small", 100.0, 2.0, 2.0, "cuda"),
        make_layout_node("tiny", 50.0, 1.0, 1.0, "cpu"),
    };

    const auto result = build_desired_layout("huge", manifest, nodes);
    if (result.success) {
        fprintf(stderr, "test-memory-fit: expected planning failure for oversized model\n");
        return 1;
    }
    if (result.layout.fits_cluster) {
        fprintf(stderr, "test-memory-fit: expected fits_cluster=false\n");
        return 1;
    }

    printf("test-memory-fit: OK rejected oversized model\n");
    return 0;
}
