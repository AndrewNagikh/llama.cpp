#include "runtime_debug/runtime_state.h"
#include "runtime_debug/hidden_transport.h"

#include <cassert>
#include <cstdio>

int main() {
    runtime_state_snapshot snap{};
    snap.step = 0;
    snap.n_past = 5;
    snap.hidden_bytes = 4096;
    assert(!runtime_state_json(snap).empty());

    const float a[] = { 1.f, 2.f, 3.f };
    const float b[] = { 1.f, 2.f, 3.f };
    assert(hidden_memcmp_buffers(a, b, 3, nullptr));

    hidden_transport_trace tr{};
    tr.payload_bytes = 12;
    tr.dtype = "f32";
    assert(!hidden_transport_trace_json(tr).empty());

    printf("test-runtime-state: OK\n");
    return 0;
}
