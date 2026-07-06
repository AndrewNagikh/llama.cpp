#include "verification/hidden_pipeline_parity.h"
#include "verification/decode_loop_parity.h"

#include <cassert>
#include <cstdio>

int main() {
    parsed_trace empty{};
    divergence_report none = find_first_divergence(empty, empty, empty, empty);
    assert(none.kind == divergence_kind::none);

    printf("test-hidden-parity: OK (framework smoke)\n");
    return 0;
}
