#include "orchestrator/layout_planner/layout_planner.h"
#include "test_layout_common.h"

#include <cstdio>
#include <map>

int main() {
    const model_manifest manifest = make_test_manifest(40, 80 * 1024 * 1024);
    const std::vector<layout_node_input> nodes = {
        make_layout_node("fast-gpu", 800.0, 8.0, 24.0, "cuda"),
        make_layout_node("slow-gpu", 200.0, 8.0, 24.0, "cuda"),
    };

    const auto result = build_desired_layout("m", manifest, nodes);
    if (!result.success) {
        fprintf(stderr, "test-gpu-priority: %s\n", result.error.c_str());
        return 1;
    }

    std::map<std::string, uint64_t> bytes_by_node;
    for (const auto & p : result.layout.placements) {
        bytes_by_node[p.node_id] += p.size_bytes;
    }

    if (bytes_by_node["fast-gpu"] <= bytes_by_node["slow-gpu"]) {
        fprintf(stderr, "test-gpu-priority: faster GPU should receive more model bytes\n");
        return 1;
    }

    printf("test-gpu-priority: OK fast=%llu slow=%llu bytes\n",
            (unsigned long long) bytes_by_node["fast-gpu"],
            (unsigned long long) bytes_by_node["slow-gpu"]);
    return 0;
}
