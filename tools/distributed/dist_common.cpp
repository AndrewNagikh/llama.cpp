#include "dist_common.h"

#include "ggml-backend.h"

#include <chrono>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <thread>

#if defined(__APPLE__)
#include <mach/mach.h>
#include <sys/sysctl.h>
#include <unistd.h>
#elif !defined(_WIN32)
#include <sys/sysinfo.h>
#include <unistd.h>
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

void dist_probe_memory(int64_t & total_mb, int64_t & free_mb) {
    total_mb = 0;
    free_mb  = 0;

#if defined(__APPLE__)
    int64_t mem_bytes = 0;
    size_t len = sizeof(mem_bytes);
    if (sysctlbyname("hw.memsize", &mem_bytes, &len, nullptr, 0) == 0) {
        total_mb = mem_bytes / (1024 * 1024);
    }

    vm_size_t page_size = 0;
    if (host_page_size(mach_host_self(), &page_size) != KERN_SUCCESS) {
        return;
    }

    vm_statistics64_data_t vm{};
    mach_msg_type_number_t count = HOST_VM_INFO64_COUNT;
    if (host_statistics64(mach_host_self(), HOST_VM_INFO64,
            reinterpret_cast<host_info64_t>(&vm), &count) == KERN_SUCCESS) {
        const int64_t free_pages = (int64_t) vm.free_count + (int64_t) vm.purgeable_count;
        free_mb = free_pages * (int64_t) page_size / (1024 * 1024);
    }
#elif !defined(_WIN32)
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

dist_node_capabilities dist_probe_capabilities() {
    dist_node_capabilities caps;
    caps.cpu_threads = (int) std::thread::hardware_concurrency();
    if (caps.cpu_threads <= 0) {
        caps.cpu_threads = 4;
    }

    ggml_backend_load_all();

    for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
        ggml_backend_dev_t dev = ggml_backend_dev_get(i);
        const auto dev_type = ggml_backend_dev_type(dev);
        if (dev_type != GGML_BACKEND_DEVICE_TYPE_GPU &&
            dev_type != GGML_BACKEND_DEVICE_TYPE_IGPU) {
            continue;
        }

        const char * desc = ggml_backend_dev_description(dev);
        caps.gpu_name = (desc && desc[0]) ? desc : ggml_backend_dev_name(dev);

        ggml_backend_reg_t reg = ggml_backend_dev_backend_reg(dev);
        if (reg) {
            const char * reg_name = ggml_backend_reg_name(reg);
            if (reg_name && reg_name[0]) {
                caps.gpu_backend = reg_name;
            }
        }

        size_t free_bytes = 0;
        size_t total_bytes = 0;
        ggml_backend_dev_memory(dev, &free_bytes, &total_bytes);
        caps.gpu_memory_mb = (int32_t) (total_bytes / (1024 * 1024));
        break;
    }

    return caps;
}

int64_t dist_now_unix() {
    return (int64_t) std::chrono::duration_cast<std::chrono::seconds>(
            std::chrono::system_clock::now().time_since_epoch()).count();
}
