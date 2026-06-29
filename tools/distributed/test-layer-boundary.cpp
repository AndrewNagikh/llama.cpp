#include "runtime_debug/layer_trace.h"
#include "runtime_debug/tensor_stats.h"
#include "verification/layer_equivalence.h"

#include <cassert>
#include <cstdio>
#include <cstdlib>

int main() {
    setenv("LLAMA_LAYER_TRACE", "1", 1);
    setenv("LLAMA_LAYER_TRACE_DIR", "/tmp/layer_boundary_test", 1);

    const layer_trace_config cfg = layer_trace_load_config();
    assert(cfg.enabled);
    assert(!cfg.trace_dir.empty());

    layer_boundary_dump dump{};
    dump.layer_index = 3;
    dump.token_index = 7;
    dump.n_embd      = 4;
    const float inp[] = { 1.f, 2.f, 3.f, 4.f };
    const float out[] = { 1.f, 2.f, 3.f, 4.f };
    dump.input        = compute_tensor_stats(inp, 4);
    dump.output       = compute_tensor_stats(out, 4);
    dump.parity       = compare_tensors(inp, out, 4);
    dump.input_ok     = true;
    dump.output_ok    = true;
    dump.positions    = { 0, 1, 2, 3, 4, 5, 6, 7 };

    layer_trace_write_boundary(cfg, "test", dump);
    assert(!layer_boundary_dump_json(dump).empty());

    layer_equivalence_report empty_report{};
    assert(!empty_report.all_pass);
    assert(layer_equivalence_report_json(empty_report).find("rows") != std::string::npos);

    printf("test-layer-boundary: OK trace_dir=%s\n", cfg.trace_dir.c_str());
    return 0;
}
