#pragma once

#include <cstdint>

// Optional GGML perf hooks (Task 12.7). No-op until callbacks are registered.

typedef void (*llama_perf_hook_fn)(const char * event);

void llama_perf_hook_set_ggml(llama_perf_hook_fn begin, llama_perf_hook_fn end);
void llama_perf_hook_ggml_begin(const char * event);
void llama_perf_hook_ggml_end(const char * event);
