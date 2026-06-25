#include "orchestrator/layout_planner/layout_planner.h"
#include "test_layout_common.h"

#include <cstdio>

int main() {
    const model_manifest manifest = make_test_manifest(32, 30 * 1024 * 1024);
    const std::vector<layout_node_input> nodes = {
        make_layout_node("a", 300.0, 8.0, 10.0, "cuda"),
        make_layout_node("b", 200.0, 8.0, 10.0, "cuda"),
        make_layout_node("c", 100.0, 8.0, 10.0, "cuda"),
    };

    const auto result = build_desired_layout("m", manifest, nodes);
    if (!result.success) {
        fprintf(stderr, "test-no-overlap: %s\n", result.error.c_str());
        return 1;
    }
    if (!layout_has_no_overlap(result.layout)) {
        fprintf(stderr, "test-no-overlap: duplicate layers detected\n");
        return 1;
    }

    printf("test-no-overlap: OK\n");
    return 0;
}
