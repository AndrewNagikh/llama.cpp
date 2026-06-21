#include "dist_common.h"

#include <cstdio>
#include <cstring>
#include <fstream>
#include <thread>

#if !defined(_WIN32)
#include <sys/sysinfo.h>
#endif

bool dist_parse_host_port(const std::string & listen, std::string & host, int & port) {
    if (listen.empty()) {
        return false;
    }

    const auto colon = listen.rfind(':');
    if (colon == std::string::npos) {
        host = "0.0.0.0";
        port = std::stoi(listen);
        return port > 0 && port < 65536;
    }

    host = listen.substr(0, colon);
    if (host.empty()) {
        host = "0.0.0.0";
    }
    port = std::stoi(listen.substr(colon + 1));
    return port > 0 && port < 65536;
}

std::string dist_role_name(dist_node_role role) {
    switch (role) {
        case DIST_ROLE_ENTRY:  return "entry";
        case DIST_ROLE_MIDDLE: return "middle";
        case DIST_ROLE_FINAL:  return "final";
        default:               return "unconfigured";
    }
}

dist_node_capabilities dist_probe_capabilities() {
    dist_node_capabilities caps;
    caps.cpu_threads = (int) std::thread::hardware_concurrency();
    if (caps.cpu_threads <= 0) {
        caps.cpu_threads = 4;
    }
    return caps;
}

void dist_probe_memory(int64_t & total_mb, int64_t & free_mb) {
    total_mb = 0;
    free_mb  = 0;

#if !defined(_WIN32)
    struct sysinfo info{};
    if (sysinfo(&info) == 0) {
        total_mb = (int64_t) info.totalram * info.mem_unit / (1024 * 1024);
        free_mb  = (int64_t) info.freeram  * info.mem_unit / (1024 * 1024);
        return;
    }

    std::ifstream f("/proc/meminfo");
    if (!f) {
        return;
    }

    std::string key;
    int64_t kb = 0;
    while (f >> key >> kb) {
        if (key == "MemTotal:") {
            total_mb = kb / 1024;
        } else if (key == "MemAvailable:") {
            free_mb = kb / 1024;
        }
    }
#endif
}
