#include "runtime_debug/perf_trace.h"

#include <chrono>
#include <cstdlib>
#include <thread>

int main() {
    setenv("DIST_PERF_TRACE", "1", 1);
    setenv("DIST_PERF_TRACE_DIR", "/tmp/dist_perf_trace_test", 1);

    if (!perf_trace_enabled()) {
        return 2;
    }

    perf_trace_set_node_id("test-node");
    perf_trace_set_component("entry");
    perf_trace_begin_generate("trace-000001", "decode");
    perf_emit_span("ENTRY_COMPUTE_END", perf_category::COMPUTE, "entry", 0, 5000, nullptr);
    perf_trace_end_generate();

    perf_trace_set_component("sync");
    perf_trace_begin_install("install-test-job-tinyllama", "install");
    perf_emit_install_span("INSTALL_BLOB", "download", "layer:0", "node-a", 1024, 5000);
    perf_trace_end_install();

    perf_trace_set_component("orchestrator");
    perf_trace_begin_session("session-test-sess1", "session");
    perf_emit_session_span("SESSION_CONFIGURE_NODE", "node-a", "entry", 230000, nullptr);
    perf_trace_end_session();

    perf_trace_begin_ttft("trace-000002", "ttft");
    perf_emit_ttft_instant("CLIENT_TTFT", "entry", "{\"token_id\":42,\"prefill_ms\":120.5}");
    std::this_thread::sleep_for(std::chrono::milliseconds(250));
    perf_trace_end_ttft();

    return 0;
}
