#include "verification/decode_loop_parity.h"
#include "verification/decode_state_parity.h"

#include <cassert>
#include <cstdio>

int main() {
    parsed_trace empty{};
    const auto report = compare_decode_state_traces(empty, empty, empty, empty);
    assert(report.all_pass);

    printf("test-decode-state-parity: OK rows=%zu\n", report.rows.size());
    return 0;
}
