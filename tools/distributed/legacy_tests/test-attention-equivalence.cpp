#include "verification/graph_equivalence.h"

#include <cassert>
#include <cstdio>
#include <string>

static bool histogram_has_op(const graph_summary & g, const std::string & op) {
    for (const auto & entry : g.op_histogram) {
        if (entry.rfind(op, 0) == 0) {
            return true;
        }
    }
    for (const auto & node : g.nodes) {
        if (node.op == op) {
            return true;
        }
    }
    return false;
}

int main() {
    graph_summary attn_block{};
    attn_block.layer_start      = 2;
    attn_block.layer_end        = 3;
    attn_block.n_compute_nodes  = 4;
    attn_block.op_histogram     = { "MUL_MAT:3", "SOFT_MAX:1", "ADD:2" };
    graph_node_info q{};
    q.op = "MUL_MAT";
    q.name = "Qcur";
    attn_block.nodes.push_back(q);

    assert(histogram_has_op(attn_block, "MUL_MAT"));
    assert(!histogram_has_op(attn_block, "ROPE"));

    graph_equivalence_result mismatch{};
    mismatch.mono   = attn_block;
    mismatch.worker = attn_block;
    mismatch.worker.n_compute_nodes = 3;
    mismatch.match = (mismatch.mono.n_compute_nodes == mismatch.worker.n_compute_nodes);
    assert(!mismatch.match);

    printf("test-attention-equivalence: OK compute_nodes=%d\n", attn_block.n_compute_nodes);
    return 0;
}
