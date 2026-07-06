#include "orchestrator/layout_planner/layout_planner.h"
#include "test_layout_common.h"

#include <cstdio>
#include <map>

int main() {
    const model_manifest manifest = make_test_manifest(8, 40 * 1024 * 1024);
    const std::vector<layout_node_input> nodes = {
        make_layout_node("node-a", 100.0, 4.0, 6.0, "cuda"),
    };

    const auto result = build_desired_layout("m1", manifest, nodes);
    if (!result.success) {
        fprintf(stderr, "test-layer-placement: %s\n", result.error.c_str());
        return 1;
    }

    std::map<int32_t, std::string> layer_nodes;
    for (const auto & p : result.layout.placements) {
        if (p.node_id.empty() || p.device.empty() || p.size_bytes == 0) {
            fprintf(stderr, "test-layer-placement: invalid placement for layer %d\n", p.layer_index);
            return 1;
        }
        layer_nodes[p.layer_index] = p.node_id;
    }
    if (static_cast<int32_t>(layer_nodes.size()) != 8) {
        fprintf(stderr, "test-layer-placement: expected 8 unique layers\n");
        return 1;
    }

    printf("test-layer-placement: OK\n");
    return 0;
}
