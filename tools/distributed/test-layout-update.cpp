#include "orchestrator/layout_planner/layout_planner.h"
#include "orchestrator/model_registry.h"
#include "test_layout_common.h"

#include <cstdio>
#include <map>
#include <string>

int main() {
    const model_manifest manifest = make_test_manifest(20, 60 * 1024 * 1024);

    cluster_model_registry reg;
    dist_model_record record;
    record.model_id = "m";
    reg.add_or_update(record);
    if (!reg.apply_manifest("m", manifest, nullptr)) {
        fprintf(stderr, "test-layout-update: apply_manifest failed\n");
        return 1;
    }

    std::vector<layout_node_input> generous = {
        make_layout_node("a", 200.0, 16.0, 16.0, "cuda"),
        make_layout_node("b", 200.0, 16.0, 16.0, "cuda"),
    };

    const auto first = build_desired_layout("m", manifest, generous);
    if (!first.success || !first.layout.fits_cluster) {
        fprintf(stderr, "test-layout-update: initial build failed: %s\n", first.error.c_str());
        return 1;
    }
    reg.apply_layout("m", first.layout, nullptr);

    std::map<std::string, int> first_counts;
    for (const auto & p : first.layout.placements) {
        first_counts[p.node_id]++;
    }

    std::vector<layout_node_input> tight = {
        make_layout_node("a", 200.0, 0.5, 0.5, "cuda"),
        make_layout_node("b", 200.0, 0.5, 0.5, "cuda"),
    };

    const auto second = build_desired_layout("m", manifest, tight);
    if (second.success) {
        fprintf(stderr, "test-layout-update: expected failure with reduced memory\n");
        return 1;
    }
    if (second.layout.fits_cluster) {
        fprintf(stderr, "test-layout-update: expected fits_cluster=false after shrink\n");
        return 1;
    }

    const auto third = build_desired_layout("m", manifest, generous);
    if (!third.success) {
        fprintf(stderr, "test-layout-update: rebuild after restore failed: %s\n", third.error.c_str());
        return 1;
    }
    reg.apply_layout("m", third.layout, nullptr);

    std::map<std::string, int> third_counts;
    for (const auto & p : third.layout.placements) {
        third_counts[p.node_id]++;
    }
    if (third_counts != first_counts) {
        fprintf(stderr, "test-layout-update: layout changed after memory restore\n");
        return 1;
    }

    printf("test-layout-update: OK\n");
    return 0;
}
