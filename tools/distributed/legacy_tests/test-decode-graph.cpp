#include "verification/decode_graph_verify.h"

#include <cassert>
#include <cstdio>

int main() {
    decode_graph_verify_result empty{};
    assert(!empty.comparison.match);
    assert(decode_graph_diff_json(empty).find("graph_kind") != std::string::npos);
    printf("test-decode-graph: OK\n");
    return 0;
}
