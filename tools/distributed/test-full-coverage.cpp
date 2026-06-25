#include "orchestrator/layout_planner/layout_planner.h"
#include "test_layout_common.h"

#include <cstdio>

int main() {
    const model_manifest manifest = make_test_manifest(24, 25 * 1024 * 1024);
    const std::vector<layout_node_input> nodes = {
        make_layout_node("a", 100.0, 6.0, 8.0, "metal"),
        make_layout_node("b", 100.0, 6.0, 8.0, "metal"),
    };

    const auto result = build_desired_layout("m", manifest, nodes);
    if (!result.success) {
        fprintf(stderr, "test-full-coverage: %s\n", result.error.c_str());
        return 1;
    }
    if (!layout_has_full_coverage(result.layout, 24)) {
        fprintf(stderr, "test-full-coverage: missing layers\n");
        return 1;
    }

    std::string err;
    if (!validate_desired_layout(result.layout, manifest, err)) {
        fprintf(stderr, "test-full-coverage: %s\n", err.c_str());
        return 1;
    }

    printf("test-full-coverage: OK\n");
    return 0;
}
