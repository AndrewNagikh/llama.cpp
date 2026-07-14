#pragma once

#include <cstddef>
#include <string>

// Resolves the path node_agent/orchestrator's launch script tees its stdout
// and stderr into (see run-agent.sh / run-orchestrator.sh). Returns "" when
// models_dir is unknown (nothing to serve).
std::string node_log_resolve_path(const std::string & models_dir, const std::string & filename);

// Returns the last max_lines lines of the file at path, scanning at most the
// final max_bytes of the file (0 = no cap). Returns "" if the file can't be
// opened.
std::string node_log_tail(const std::string & path, size_t max_lines, size_t max_bytes);
