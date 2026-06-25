#include "dist_common.h"

#include "ggml-backend.h"

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <set>
#include <sstream>
#include <thread>

#if defined(__APPLE__)
#include <mach/mach.h>
#include <sys/sysctl.h>
#include <unistd.h>
#elif defined(_WIN32)
// clang-format off
#include <windows.h>
#include <sysinfoapi.h>
#include <intrin.h>
// clang-format on
#else
#include <sys/sysinfo.h>
#include <unistd.h>
#endif

#ifdef _WIN32
static std::string dist_cpu_brand_string_win() {
    int brand[4 * 4] = { 0 };
    char buf[64] = { 0 };
    __cpuid((int *) brand, 0x80000000);
    const unsigned int max_ext = static_cast<unsigned int>(brand[0]);
    if (max_ext < 0x80000004) {
        return "";
    }
    int * p = brand;
    for (unsigned int f = 0x80000002; f <= 0x80000004; ++f) {
        __cpuid(p, (int) f);
        p += 4;
    }
    return std::string((const char *) brand);
}
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

double dist_bytes_to_gb(const uint64_t bytes) {
    return static_cast<double>(bytes) / (1024.0 * 1024.0 * 1024.0);
}

void dist_probe_node_memory(dist_node_memory & out) {
    out.total_ram_bytes  = 0;
    out.free_ram_bytes   = 0;
    out.total_vram_bytes = 0;
    out.free_vram_bytes  = 0;
    out.has_gpu          = false;

#if defined(__APPLE__)
    int64_t mem_bytes = 0;
    size_t len = sizeof(mem_bytes);
    if (sysctlbyname("hw.memsize", &mem_bytes, &len, nullptr, 0) == 0) {
        out.total_ram_bytes = static_cast<uint64_t>(mem_bytes);
    }

    vm_size_t page_size = 0;
    if (host_page_size(mach_host_self(), &page_size) == KERN_SUCCESS) {
        vm_statistics64_data_t vm{};
        mach_msg_type_number_t count = HOST_VM_INFO64_COUNT;
        if (host_statistics64(mach_host_self(), HOST_VM_INFO64,
                reinterpret_cast<host_info64_t>(&vm), &count) == KERN_SUCCESS) {
            const uint64_t ps = static_cast<uint64_t>(page_size);

            // Do not use vm.free_count alone: on macOS it is only the tiny pool of
            // completely unused pages. Most reclaimable memory lives in inactive and
            // purgeable caches. The formula below matches macOS memory_pressure's
            // "system-wide memory free percentage" (total minus wired/compressed).
            const uint64_t pinned_pages = static_cast<uint64_t>(vm.wire_count) +
                                          static_cast<uint64_t>(vm.compressor_page_count);
            if (out.total_ram_bytes > pinned_pages * ps) {
                out.free_ram_bytes = out.total_ram_bytes - pinned_pages * ps;
            } else {
                const uint64_t reclaimable_pages = static_cast<uint64_t>(vm.free_count) +
                                                   static_cast<uint64_t>(vm.inactive_count) +
                                                   static_cast<uint64_t>(vm.speculative_count) +
                                                   static_cast<uint64_t>(vm.purgeable_count);
                out.free_ram_bytes = reclaimable_pages * ps;
            }
        }
    }
#elif defined(_WIN32)
    MEMORYSTATUSEX status{};
    status.dwLength = sizeof(status);
    if (GlobalMemoryStatusEx(&status)) {
        out.total_ram_bytes = status.ullTotalPhys;
        out.free_ram_bytes  = status.ullAvailPhys;
    }
#else
    // Prefer MemAvailable from /proc/meminfo; it includes reclaimable cache.
    {
        std::ifstream f("/proc/meminfo");
        if (f) {
            std::string key;
            uint64_t kb = 0;
            while (f >> key >> kb) {
                if (key == "MemTotal:") {
                    out.total_ram_bytes = kb * 1024;
                } else if (key == "MemAvailable:") {
                    out.free_ram_bytes = kb * 1024;
                }
                std::string rest;
                std::getline(f, rest);
            }
        }
    }
    if (out.total_ram_bytes == 0 || out.free_ram_bytes == 0) {
        struct sysinfo info{};
        if (sysinfo(&info) == 0) {
            out.total_ram_bytes = static_cast<uint64_t>(info.totalram) * info.mem_unit;
            out.free_ram_bytes  = static_cast<uint64_t>(info.freeram)  * info.mem_unit;
        }
    }
#endif

    // GPU memory via ggml backend devices.
    ggml_backend_load_all();
    for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
        ggml_backend_dev_t dev = ggml_backend_dev_get(i);
        const auto dev_type = ggml_backend_dev_type(dev);
        if (dev_type != GGML_BACKEND_DEVICE_TYPE_GPU &&
            dev_type != GGML_BACKEND_DEVICE_TYPE_IGPU) {
            continue;
        }

        size_t free_bytes = 0;
        size_t total_bytes = 0;
        ggml_backend_dev_memory(dev, &free_bytes, &total_bytes);
        out.total_vram_bytes += total_bytes;
        out.free_vram_bytes  += free_bytes;
        out.has_gpu           = true;
    }
}

#if !defined(_WIN32)
static std::string dist_read_proc_cpuinfo_field(const char * field) {
    std::ifstream f("/proc/cpuinfo");
    if (!f) {
        return "";
    }
    std::string line;
    const size_t n = std::strlen(field);
    while (std::getline(f, line)) {
        if (line.size() >= n && std::strncmp(line.c_str(), field, n) == 0) {
            const auto pos = line.find(':');
            if (pos != std::string::npos) {
                std::string value = line.substr(pos + 1);
                // Trim leading spaces.
                const auto start = value.find_first_not_of(" \t\r\n");
                if (start == std::string::npos) {
                    return "";
                }
                return value.substr(start);
            }
        }
    }
    return "";
}
#endif

void dist_probe_node_cpu(dist_node_cpu & out) {
    out.physical_cores = 0;
    out.logical_cores  = 0;
    out.cache_l3_bytes = 0;
    out.cpu_name.clear();

    int logical = static_cast<int>(std::thread::hardware_concurrency());
    if (logical > 0) {
        out.logical_cores = logical;
    }

#if defined(__APPLE__)
    size_t len = 0;
    char name[256] = { 0 };
    len = sizeof(name);
    if (sysctlbyname("machdep.cpu.brand_string", name, &len, nullptr, 0) == 0) {
        out.cpu_name = std::string(name);
    }
    int phys = 0;
    len = sizeof(phys);
    if (sysctlbyname("hw.physicalcpu", &phys, &len, nullptr, 0) == 0 && phys > 0) {
        out.physical_cores = phys;
    }
    int logi = 0;
    len = sizeof(logi);
    if (sysctlbyname("hw.logicalcpu", &logi, &len, nullptr, 0) == 0 && logi > 0) {
        out.logical_cores = logi;
    }
    int64_t l3 = 0;
    len = sizeof(l3);
    if (sysctlbyname("hw.l3cachesize", &l3, &len, nullptr, 0) == 0) {
        out.cache_l3_bytes = static_cast<uint64_t>(l3);
    }
#elif defined(_WIN32)
    out.cpu_name = dist_cpu_brand_string_win();
    SYSTEM_INFO si{};
    GetSystemInfo(&si);
    out.physical_cores = static_cast<int>(si.dwNumberOfProcessors);
    if (out.logical_cores <= 0) {
        out.logical_cores = out.physical_cores;
    }
#else
    long nprocs = sysconf(_SC_NPROCESSORS_ONLN);
    if (nprocs > 0) {
        out.logical_cores = static_cast<int>(nprocs);
    }
    out.cpu_name = dist_read_proc_cpuinfo_field("model name");

    std::ifstream f("/proc/cpuinfo");
    if (f) {
        std::string line;
        std::set<int> physical_ids;
        while (std::getline(f, line)) {
            if (line.find("physical id") != std::string::npos) {
                const auto pos = line.find(':');
                if (pos != std::string::npos) {
                    physical_ids.insert(std::stoi(line.substr(pos + 1)));
                }
            }
        }
        if (!physical_ids.empty()) {
            out.physical_cores = static_cast<int>(physical_ids.size());
        }
    }
    if (out.physical_cores <= 0 && out.logical_cores > 0) {
        out.physical_cores = out.logical_cores;
    }
#endif
}

dist_node_system dist_probe_node_system() {
    dist_node_system sys;
#if defined(_WIN32)
    sys.os = "windows";
#elif defined(__APPLE__)
    sys.os = "macos";
#else
    sys.os = "linux";
#endif

#if defined(__x86_64__) || defined(_M_X64)
    sys.arch = "x86_64";
#elif defined(__aarch64__) || defined(_M_ARM64)
    sys.arch = "arm64";
#elif defined(__arm__) || defined(_M_ARM)
    sys.arch = "arm";
#elif defined(__i386__) || defined(_M_IX86)
    sys.arch = "x86";
#else
    sys.arch = "unknown";
#endif
    return sys;
}

// Legacy probes (kept for compatibility with Task 5/6 code).
void dist_probe_memory(int64_t & total_mb, int64_t & free_mb) {
    dist_node_memory mem{};
    dist_probe_node_memory(mem);
    total_mb = static_cast<int64_t>(mem.total_ram_bytes / (1024 * 1024));
    free_mb  = static_cast<int64_t>(mem.free_ram_bytes  / (1024 * 1024));
}

dist_node_capabilities dist_probe_capabilities() {
    dist_node_capabilities caps;
    caps.cpu_threads = static_cast<int>(std::thread::hardware_concurrency());
    if (caps.cpu_threads <= 0) {
        caps.cpu_threads = 4;
    }

    dist_node_memory mem{};
    dist_probe_node_memory(mem);
    caps.total_ram_bytes  = mem.total_ram_bytes;
    caps.free_ram_bytes   = mem.free_ram_bytes;
    caps.total_vram_bytes = mem.total_vram_bytes;
    caps.free_vram_bytes  = mem.free_vram_bytes;
    caps.has_gpu          = mem.has_gpu;

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
        caps.gpu_memory_mb = static_cast<int32_t>(total_bytes / (1024 * 1024));
        break;
    }

    dist_node_cpu cpu{};
    dist_probe_node_cpu(cpu);
    caps.cpu_name = cpu.cpu_name;
    caps.cpu_threads = cpu.logical_cores > 0 ? cpu.logical_cores : caps.cpu_threads;

    caps.os   = dist_probe_node_system().os;
    caps.arch = dist_probe_node_system().arch;

    return caps;
}

std::string dist_normalize_device(const std::string & backend, bool has_gpu) {
    if (!has_gpu) {
        return "cpu";
    }
    std::string b;
    b.reserve(backend.size());
    for (char c : backend) {
        b += static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    }
    if (b.find("metal") != std::string::npos) {
        return "metal";
    }
    if (b.find("cuda") != std::string::npos ||
        b.find("vulkan") != std::string::npos ||
        b.find("hip") != std::string::npos ||
        b.find("rocm") != std::string::npos ||
        b.find("musa") != std::string::npos) {
        return "cuda";
    }
    if (b.find("gpu") != std::string::npos) {
        return "cuda";
    }
    return "cpu";
}

int64_t dist_now_unix() {
    return static_cast<int64_t>(std::chrono::duration_cast<std::chrono::seconds>(
            std::chrono::system_clock::now().time_since_epoch()).count());
}
