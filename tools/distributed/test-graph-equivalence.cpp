#include "verification/graph_equivalence.h"

#include <cassert>
#include <cstdio>

int main() {
    graph_summary empty{};
    assert(empty.n_nodes == 0);
    assert(!graph_summary_json(empty).empty());

    graph_equivalence_result no_ctx{};
    assert(!no_ctx.match);

    graph_summary a{};
    a.layer_start = 0;
    a.layer_end   = 1;
    a.n_nodes     = 2;
    a.n_compute_nodes = 1;
    graph_node_info n{};
    n.op = "ROPE";
    a.nodes.push_back(n);
    a.op_histogram.push_back("ROPE:1");

    graph_summary b = a;
    graph_equivalence_result same{};
    same.mono   = a;
    same.worker = b;
    same.match  = (a.n_nodes == b.n_nodes);

    assert(same.match);
    printf("test-graph-equivalence: OK nodes=%d\n", a.n_nodes);
    return 0;
}
