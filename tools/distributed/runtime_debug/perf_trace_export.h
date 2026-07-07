#pragma once

#include <string>

#include <nlohmann/json.hpp>

std::string perf_trace_resolve_root(const std::string & models_dir);

bool perf_trace_safe_rel(const std::string & rel);

nlohmann::json perf_trace_list_files(const std::string & root);

bool perf_trace_read_file(const std::string & root, const std::string & rel, std::string & out);
