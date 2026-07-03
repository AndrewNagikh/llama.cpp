#include "dist_rss_probe.h"

#include <cstdio>

#if defined(__linux__)
#include <fstream>
#include <string>
#elif defined(__APPLE__)
#include <mach/mach.h>
#endif

static uint64_t g_rss_baseline = 0;

uint64_t dist_process_rss_bytes() {
#if defined(__linux__)
    std::ifstream status("/proc/self/status");
    if (!status) {
        return 0;
    }
    std::string line;
    while (std::getline(status, line)) {
        if (line.rfind("VmRSS:", 0) == 0) {
            unsigned long kb = 0;
            if (std::sscanf(line.c_str(), "VmRSS: %lu", &kb) == 1) {
                return static_cast<uint64_t>(kb) * 1024ULL;
            }
        }
    }
    return 0;
#elif defined(__APPLE__)
    mach_task_basic_info info{};
    mach_msg_type_number_t count = MACH_TASK_BASIC_INFO_COUNT;
    if (task_info(mach_task_self(), MACH_TASK_BASIC_INFO,
            reinterpret_cast<task_info_t>(&info), &count) == KERN_SUCCESS) {
        return static_cast<uint64_t>(info.resident_size);
    }
    return 0;
#else
    return 0;
#endif
}

void dist_rss_set_baseline(const uint64_t bytes) {
    g_rss_baseline = bytes;
}

uint64_t dist_rss_baseline_bytes() {
    return g_rss_baseline;
}

static double bytes_to_mb(const uint64_t bytes) {
    return static_cast<double>(bytes) / (1024.0 * 1024.0);
}

void dist_rss_log_stage(const char * stage, const char * detail) {
    const uint64_t rss = dist_process_rss_bytes();
    const double delta_mb =
            (static_cast<int64_t>(rss) - static_cast<int64_t>(g_rss_baseline)) / (1024.0 * 1024.0);
    if (detail != nullptr && detail[0] != '\0') {
        fprintf(stderr,
                "orchestrator: RSS stage=%s model=%s rss=%.1f MB baseline_delta=%+.1f MB\n",
                stage, detail, bytes_to_mb(rss), delta_mb);
    } else {
        fprintf(stderr,
                "orchestrator: RSS stage=%s rss=%.1f MB baseline_delta=%+.1f MB\n",
                stage, bytes_to_mb(rss), delta_mb);
    }
}

dist_rss_scope::dist_rss_scope(const char * stage, const char * detail)
        : stage_(stage ? stage : ""), detail_(detail ? detail : ""), rss_enter_(dist_process_rss_bytes()) {
    dist_rss_log_stage((stage_ + " enter").c_str(), detail_.empty() ? nullptr : detail_.c_str());
}

dist_rss_scope::~dist_rss_scope() {
    const uint64_t rss_exit = dist_process_rss_bytes();
    const double scope_delta_mb =
            static_cast<double>(static_cast<int64_t>(rss_exit) - static_cast<int64_t>(rss_enter_)) /
            (1024.0 * 1024.0);
    fprintf(stderr,
            "orchestrator: RSS stage=%s leave model=%s rss=%.1f MB scope_delta=%+.1f MB\n",
            stage_.c_str(),
            detail_.empty() ? "-" : detail_.c_str(),
            bytes_to_mb(rss_exit),
            scope_delta_mb);
}
