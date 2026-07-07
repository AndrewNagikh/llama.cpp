#include "perf_trace_export.h"

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <sstream>

namespace fs = std::filesystem;

std::string perf_trace_resolve_root(const std::string & models_dir) {
    if (const char * dir = std::getenv("DIST_PERF_TRACE_DIR")) {
        if (*dir) {
            return std::string(dir);
        }
    }
    if (!models_dir.empty()) {
        return models_dir + "/perf_trace";
    }
    return {};
}

bool perf_trace_safe_rel(const std::string & rel) {
    if (rel.empty() || rel.front() == '/') {
        return false;
    }
    return rel.find("..") == std::string::npos;
}

nlohmann::json perf_trace_list_files(const std::string & root) {
    nlohmann::json files = nlohmann::json::array();
    if (root.empty() || !fs::exists(root)) {
        return files;
    }

    std::error_code ec;
    for (const auto & ent : fs::recursive_directory_iterator(root, ec)) {
        if (ec) {
            break;
        }
        if (!ent.is_regular_file()) {
            continue;
        }
        const fs::path & p = ent.path();
        if (p.extension() != ".jsonl") {
            continue;
        }
        const std::string rel = fs::relative(p, root, ec).generic_string();
        if (ec || rel.empty()) {
            continue;
        }
        files.push_back({
            { "rel",  rel },
            { "size", fs::file_size(p, ec) },
        });
    }
    return files;
}

bool perf_trace_read_file(const std::string & root, const std::string & rel, std::string & out) {
    if (!perf_trace_safe_rel(rel) || root.empty()) {
        return false;
    }
    const fs::path full = fs::path(root) / rel;
    std::ifstream in(full, std::ios::binary);
    if (!in) {
        return false;
    }
    std::ostringstream ss;
    ss << in.rdbuf();
    out = ss.str();
    return true;
}
