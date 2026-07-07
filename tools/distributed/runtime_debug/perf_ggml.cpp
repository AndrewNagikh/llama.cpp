#include "perf_ggml.h"

#include "perf_trace.h"

#include "llama_perf_hooks.h"

#include <cstdlib>
#include <cstring>
#include <map>
#include <mutex>

namespace {

static bool env_truthy(const char * name) {
    const char * v = std::getenv(name);
    if (v == nullptr || v[0] == '\0') {
        return false;
    }
    return std::strcmp(v, "0") != 0 &&
           std::strcmp(v, "false") != 0 &&
           std::strcmp(v, "FALSE") != 0;
}

static thread_local std::map<std::string, int64_t> g_ggml_t0;
static std::mutex g_reg_mu;
static bool       g_hooks_registered = false;

static const char * current_stage() {
    const std::string & comp = perf_trace_config_get().component;
    if (comp == "entry" || comp == "middle" || comp == "final") {
        return comp.c_str();
    }
    return "node";
}

static void ggml_hook_begin(const char * event) {
    if (!perf_trace_ggml_enabled()) {
        return;
    }
    perf_trace_refresh_context();
    g_ggml_t0[event ? event : ""] = static_cast<int64_t>(perf_now_us());
}

static void ggml_hook_end(const char * event) {
    if (!perf_trace_ggml_enabled()) {
        return;
    }
    perf_trace_refresh_context();
    const std::string key = event ? event : "";
    const auto it = g_ggml_t0.find(key);
    if (it == g_ggml_t0.end()) {
        return;
    }
    const int64_t dur = static_cast<int64_t>(perf_now_us()) - it->second;
    g_ggml_t0.erase(it);

    perf_category cat = perf_category::COMPUTE;
    if (key.find("SYNC") != std::string::npos || key.find("SCHED_") == 0) {
        cat = perf_category::WAIT;
    }

    int32_t token_idx = -1;
    std::string trace_id;
    std::string phase;
    perf_trace_get_context(trace_id, phase, token_idx);

    perf_emit_span(key.c_str(), cat, current_stage(), token_idx, dur, nullptr);
}

} // namespace

bool perf_trace_ggml_enabled() {
    return perf_trace_enabled() && env_truthy("DIST_PERF_TRACE_GGML");
}

void perf_trace_register_ggml_hooks() {
    std::lock_guard<std::mutex> lock(g_reg_mu);
    if (g_hooks_registered) {
        return;
    }
    llama_perf_hook_set_ggml(ggml_hook_begin, ggml_hook_end);
    g_hooks_registered = true;
}

void perf_trace_sched_queue_wait_begin() {
    if (!perf_trace_ggml_enabled()) {
        return;
    }
    llama_perf_hook_ggml_begin("SCHED_QUEUE_WAIT");
}

void perf_trace_sched_queue_wait_end() {
    if (!perf_trace_ggml_enabled()) {
        return;
    }
    llama_perf_hook_ggml_end("SCHED_QUEUE_WAIT");
}
