#include "orchestrator/layout_planner/layout_planner.h"
#include "orchestrator/model_registry.h"
#include "test_layout_common.h"

#include <cstdio>
#include <string>

int main() {
    const model_manifest manifest = make_test_manifest(30, 100 * 1024 * 1024);

    cluster_model_registry reg;
    dist_model_record record;
    record.model_id = "m";
    reg.add_or_update(record);
    if (!reg.apply_manifest("m", manifest, nullptr)) {
        fprintf(stderr, "test-layout-node-loss: apply_manifest failed\n");
        return 1;
    }

    const std::vector<layout_node_input> three_nodes = {
        make_layout_node("a", 300.0, 1.5, 1.5, "cuda"),
        make_layout_node("b", 200.0, 1.5, 1.5, "cuda"),
        make_layout_node("c", 100.0, 1.5, 1.5, "cuda"),
    };

    const auto with_three = build_desired_layout("m", manifest, three_nodes);
    if (!with_three.success || !with_three.layout.fits_cluster) {
        fprintf(stderr, "test-layout-node-loss: initial build failed: %s\n", with_three.error.c_str());
        return 1;
    }

    const std::vector<layout_node_input> two_nodes = {
        make_layout_node("a", 300.0, 1.5, 1.5, "cuda"),
        make_layout_node("b", 200.0, 1.5, 1.5, "cuda"),
    };

    const auto after_loss = build_desired_layout("m", manifest, two_nodes);
    if (after_loss.success) {
        fprintf(stderr, "test-layout-node-loss: expected planning failure after node loss\n");
        return 1;
    }
    if (after_loss.layout.fits_cluster) {
        fprintf(stderr, "test-layout-node-loss: expected fits_cluster=false\n");
        return 1;
    }

    printf("test-layout-node-loss: OK fits_cluster=false after node loss\n");
    return 0;
}
