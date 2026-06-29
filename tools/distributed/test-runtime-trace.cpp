#include "runtime_debug/tensor_stats.h"
#include "runtime_debug/trace_recorder.h"
#include "runtime_debug/runtime_debug.h"
#include "verification/decode_loop_parity.h"

#include <cassert>
#include <cstdio>
#include <filesystem>
#include <fstream>

static void write_temp_trace(const std::string & path, const std::string & body) {
    std::ofstream out(path);
    out << body << '\n';
}

int main() {
    setenv("LLAMA_DISTRIBUTED_DEBUG", "1", 1);
    setenv("LLAMA_DIST_TRACE_DIR", "/tmp/dist_trace_test", 1);
    dist_debug_set_session("test_sess");
    dist_debug_set_worker("entry", "node-a");
    dist_debug_reset_recorder("entry");

    trace_recorder * rec = dist_debug_recorder();
    assert(rec != nullptr);

    const float hidden[] = { 1.0f, -2.0f, 3.0f, 4.0f };
    rec->emit_step_begin(0, "prefill", -1, 0, 0);
    rec->emit_hidden(0, "prefill", hidden, 1, 4, "test");
    rec->emit_token_selected(0, "prefill", 42, 5, false);

    const std::string trace_path = rec->trace_path();
    assert(!trace_path.empty());
    assert(std::filesystem::exists(trace_path));

    const parsed_trace parsed = parse_trace_jsonl(trace_path);
    assert(parsed.events.size() >= 2);

    const float a[] = { 1.f, 2.f, 3.f };
    assert(compare_tensors(a, a, 3).match);
    assert(!compare_tensors(a, (const float[]) { 1.f, 2.f, 9.f }, 3).match);

    tensor_stats stats = compute_tensor_stats(a, 3);
    assert(stats.n_elements == 3);
    assert(!stats.sha256.empty());

    printf("test-runtime-trace: OK events=%zu sha256=%s\n",
            parsed.events.size(), stats.sha256.substr(0, 12).c_str());
    return 0;
}
