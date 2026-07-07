#include "llama_perf_hooks.h"

#include <atomic>

namespace {

std::atomic<llama_perf_hook_fn> g_ggml_begin{nullptr};
std::atomic<llama_perf_hook_fn> g_ggml_end{nullptr};

} // namespace

void llama_perf_hook_set_ggml(const llama_perf_hook_fn begin, const llama_perf_hook_fn end) {
    g_ggml_begin.store(begin, std::memory_order_relaxed);
    g_ggml_end.store(end, std::memory_order_relaxed);
}

void llama_perf_hook_ggml_begin(const char * event) {
    const auto fn = g_ggml_begin.load(std::memory_order_relaxed);
    if (fn != nullptr) {
        fn(event);
    }
}

void llama_perf_hook_ggml_end(const char * event) {
    const auto fn = g_ggml_end.load(std::memory_order_relaxed);
    if (fn != nullptr) {
        fn(event);
    }
}
