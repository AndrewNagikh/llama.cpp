#pragma once

#include <cstdint>
#include <string>

#include <nlohmann/json.hpp>

std::string perf_trace_resolve_root(const std::string & models_dir);

bool perf_trace_safe_rel(const std::string & rel);

nlohmann::json perf_trace_list_files(const std::string & root);

bool perf_trace_read_file(const std::string & root, const std::string & rel, std::string & out);

struct perf_trace_cleanup_result {
    int64_t deleted_files = 0;
    int64_t freed_bytes   = 0;
};

// Deletes *.jsonl files under root whose mtime is older than max_age_days.
// No-op (returns a zeroed result) if root is empty/missing or max_age_days <= 0.
perf_trace_cleanup_result perf_trace_cleanup(const std::string & root, int max_age_days);
