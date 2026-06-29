#pragma once

#include <cstdint>
#include <string>

// Task 9.8.3 — Distributed Runtime Debug Framework
// Observation-only hooks. Does not alter inference when disabled.

struct dist_debug_config {
    bool enabled       = false;
    bool skip_sampler  = false;
    bool dump_raw_bins = false;
    std::string trace_dir;
    std::string session_id;
    std::string worker_id;
    std::string node_id;
};

dist_debug_config dist_debug_load_config();
bool dist_debug_enabled();
bool dist_debug_env_truthy(const char * name);
bool dist_debug_skip_sampler();
const dist_debug_config & dist_debug_config_get();

void dist_debug_set_session(const std::string & session_id);
void dist_debug_set_worker(const std::string & worker_id, const std::string & node_id);

uint64_t dist_debug_now_ms();
