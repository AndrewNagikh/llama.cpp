#include "perf_trace_export.h"

#include <chrono>
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
        int64_t mtime_unix = 0;
        std::error_code mec;
        const auto mtime = fs::last_write_time(p, mec);
        if (!mec) {
            // file_time_type epoch differs from system_clock's pre-C++20; convert
            // through the clocks' current instants.
            const auto sys_now = std::chrono::system_clock::now();
            const auto fs_now  = fs::file_time_type::clock::now();
            const auto sys_mtime = sys_now + std::chrono::duration_cast<std::chrono::system_clock::duration>(mtime - fs_now);
            mtime_unix = std::chrono::duration_cast<std::chrono::seconds>(sys_mtime.time_since_epoch()).count();
        }
        files.push_back({
            { "rel",        rel },
            { "size",       fs::file_size(p, ec) },
            { "mtime_unix", mtime_unix },
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

perf_trace_cleanup_result perf_trace_cleanup(const std::string & root, const int max_age_days) {
    perf_trace_cleanup_result result{};
    if (root.empty() || max_age_days <= 0) {
        return result;
    }
    std::error_code ec;
    if (!fs::exists(root, ec) || ec) {
        return result;
    }

    // Compare within filesystem-clock's own domain -- avoids the C++17
    // file_time_type -> system_clock conversion dance entirely.
    const auto now_fs = fs::file_time_type::clock::now();
    const auto cutoff  = now_fs - std::chrono::hours(24 * (int64_t) max_age_days);

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
        std::error_code fec;
        const auto mtime = fs::last_write_time(p, fec);
        if (fec || mtime >= cutoff) {
            continue;
        }
        const auto size = fs::file_size(p, fec);
        if (fs::remove(p, fec)) {
            result.deleted_files++;
            result.freed_bytes += fec ? 0 : (int64_t) size;
        }
    }
    return result;
}
