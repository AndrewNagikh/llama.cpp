#pragma once

// Task 12.6 — background GPU/CPU utilization sampling for perf trace.
// Poll interval: DIST_PERF_GPU_POLL_MS (default 100).

void perf_gpu_poll_start();
void perf_gpu_poll_stop();
