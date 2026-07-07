#pragma once

bool perf_trace_ggml_enabled();
void perf_trace_register_ggml_hooks();
void perf_trace_sched_queue_wait_begin();
void perf_trace_sched_queue_wait_end();
