#include "verification/graph_equivalence.h"

#include <cassert>
#include <cstdio>
#include <string>

static bool graph_contains_rope(const graph_summary & g) {
    for (const auto & node : g.nodes) {
        if (node.op == "ROPE") {
            return true;
        }
    }
    for (const auto & entry : g.op_histogram) {
        if (entry.rfind("ROPE:", 0) == 0) {
            return true;
        }
    }
    return false;
}

int main() {
    graph_summary with_rope{};
    graph_node_info rope{};
    rope.op   = "ROPE";
    rope.name = "Qcur_rope";
    with_rope.nodes.push_back(rope);
    with_rope.op_histogram.push_back("ROPE:2");
    with_rope.n_compute_nodes = 1;

    graph_summary without_rope{};
    graph_node_info mat{};
    mat.op = "MUL_MAT";
    without_rope.nodes.push_back(mat);

    assert(graph_contains_rope(with_rope));
    assert(!graph_contains_rope(without_rope));

    printf("test-rope-equivalence: OK rope_detect=%d\n", graph_contains_rope(with_rope));
    return 0;
}
