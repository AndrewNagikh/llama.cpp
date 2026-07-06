#include "verification/runtime_cache_verify.h"

#include <cassert>
#include <cstdio>

int main() {
    runtime_cache_diag empty{};
    assert(empty.message.empty() || !empty.same_graph_ptr);
    assert(runtime_cache_diag_json(empty).find("same_node_count") != std::string::npos);
    printf("test-runtime-cache: OK\n");
    return 0;
}
