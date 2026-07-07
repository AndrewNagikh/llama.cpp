#include "perf_gpu_sampler.h"

#include "perf_trace.h"

#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <thread>

#if defined(__linux__) || defined(__APPLE__)
#include <dlfcn.h>
#endif

#if defined(__linux__) || defined(__APPLE__) || defined(_WIN32)
#include <sys/time.h>
#endif

#if defined(__APPLE__)
#include <mach/mach.h>
#include <sys/resource.h>
#elif defined(__linux__)
#include <sys/resource.h>
#endif

namespace {

using clock = std::chrono::steady_clock;

struct gpu_sample_reading {
    const char * backend = "cpu";
    float        util_pct = 0.0f;
    float        mem_used_mb = 0.0f;
    float        cpu_busy_pct = 0.0f;
    bool         util_valid = false;
};

struct cpu_usage_tracker {
    uint64_t last_wall_us = 0;
    uint64_t last_cpu_us  = 0;
    bool     has_prev     = false;

    static uint64_t thread_cpu_us() {
#if defined(__APPLE__)
        thread_basic_info_data_t info{};
        mach_msg_type_number_t count = THREAD_BASIC_INFO_COUNT;
        if (thread_info(mach_thread_self(), THREAD_BASIC_INFO,
                reinterpret_cast<thread_info_t>(&info), &count) != KERN_SUCCESS) {
            return 0;
        }
        return static_cast<uint64_t>(info.user_time.seconds) * 1000000ULL
             + static_cast<uint64_t>(info.user_time.microseconds)
             + static_cast<uint64_t>(info.system_time.seconds) * 1000000ULL
             + static_cast<uint64_t>(info.system_time.microseconds);
#else
        struct rusage ru {};
        if (getrusage(RUSAGE_THREAD, &ru) != 0) {
            if (getrusage(RUSAGE_SELF, &ru) != 0) {
                return 0;
            }
        }
        return static_cast<uint64_t>(ru.ru_utime.tv_sec) * 1000000ULL
             + static_cast<uint64_t>(ru.ru_utime.tv_usec)
             + static_cast<uint64_t>(ru.ru_stime.tv_sec) * 1000000ULL
             + static_cast<uint64_t>(ru.ru_stime.tv_usec);
#endif
    }

    float sample_busy_pct() {
        const uint64_t wall_us = static_cast<uint64_t>(
                std::chrono::duration_cast<std::chrono::microseconds>(
                        clock::now().time_since_epoch()).count());
        const uint64_t cpu_us = thread_cpu_us();
        float busy = 0.0f;
        if (has_prev && wall_us > last_wall_us) {
            const uint64_t d_wall = wall_us - last_wall_us;
            const uint64_t d_cpu  = cpu_us >= last_cpu_us ? cpu_us - last_cpu_us : 0;
            busy = static_cast<float>(100.0 * static_cast<double>(d_cpu) / static_cast<double>(d_wall));
            if (!std::isfinite(busy)) {
                busy = 0.0f;
            }
            if (busy < 0.0f) {
                busy = 0.0f;
            }
            if (busy > 100.0f) {
                busy = 100.0f;
            }
        }
        last_wall_us = wall_us;
        last_cpu_us  = cpu_us;
        has_prev     = true;
        return busy;
    }
};

#if defined(__linux__) || defined(__APPLE__)
struct nvml_utilization_t {
    unsigned int gpu;
    unsigned int memory;
};

struct nvml_memory_t {
    unsigned long long total;
    unsigned long long free;
    unsigned long long used;
};

struct nvml_sampler {
    void * lib = nullptr;
    void * device = nullptr;
    using init_fn = int (*)();
    using shutdown_fn = int (*)();
    using get_handle_fn = int (*)(unsigned int, void **);
    using get_util_fn = int (*)(void *, nvml_utilization_t *);
    using get_mem_fn = int (*)(void *, nvml_memory_t *);
    init_fn init = nullptr;
    shutdown_fn shutdown = nullptr;
    get_handle_fn get_handle = nullptr;
    get_util_fn get_util = nullptr;
    get_mem_fn get_mem = nullptr;
    bool ready = false;

    bool try_load() {
        if (lib != nullptr) {
            return ready;
        }
        static const char * libs[] = {
            "libnvidia-ml.so.1",
            "libnvidia-ml.so",
            nullptr,
        };
        for (const char * const name : libs) {
            if (name == nullptr) {
                break;
            }
            lib = dlopen(name, RTLD_LAZY | RTLD_LOCAL);
            if (lib != nullptr) {
                break;
            }
        }
        if (lib == nullptr) {
            return false;
        }
        init       = reinterpret_cast<init_fn>(dlsym(lib, "nvmlInit_v2"));
        if (init == nullptr) {
            init = reinterpret_cast<init_fn>(dlsym(lib, "nvmlInit"));
        }
        shutdown   = reinterpret_cast<shutdown_fn>(dlsym(lib, "nvmlShutdown"));
        get_handle = reinterpret_cast<get_handle_fn>(dlsym(lib, "nvmlDeviceGetHandleByIndex_v2"));
        if (get_handle == nullptr) {
            get_handle = reinterpret_cast<get_handle_fn>(dlsym(lib, "nvmlDeviceGetHandleByIndex"));
        }
        get_util = reinterpret_cast<get_util_fn>(dlsym(lib, "nvmlDeviceGetUtilizationRates"));
        get_mem  = reinterpret_cast<get_mem_fn>(dlsym(lib, "nvmlDeviceGetMemoryInfo"));
        if (init == nullptr || get_handle == nullptr || get_util == nullptr) {
            dlclose(lib);
            lib = nullptr;
            return false;
        }
        if (init() != 0) {
            dlclose(lib);
            lib = nullptr;
            return false;
        }
        if (get_handle(0, &device) != 0) {
            if (shutdown != nullptr) {
                shutdown();
            }
            dlclose(lib);
            lib = nullptr;
            device = nullptr;
            return false;
        }
        ready = true;
        return true;
    }

    bool sample(gpu_sample_reading & out) {
        if (!try_load() || device == nullptr || get_util == nullptr) {
            return false;
        }
        nvml_utilization_t util {};
        if (get_util(device, &util) != 0) {
            return false;
        }
        out.backend = "cuda";
        out.util_pct = static_cast<float>(util.gpu);
        out.util_valid = true;
        if (get_mem != nullptr) {
            nvml_memory_t mem {};
            if (get_mem(device, &mem) == 0) {
                out.mem_used_mb = static_cast<float>(mem.used) / (1024.0f * 1024.0f);
            }
        }
        return true;
    }

    ~nvml_sampler() {
        if (ready && shutdown != nullptr) {
            shutdown();
        }
        if (lib != nullptr) {
            dlclose(lib);
        }
    }
};
#endif

static std::mutex              g_poll_mu;
static std::thread             g_poll_thread;
static std::atomic<int>        g_poll_refs{0};
static std::atomic<bool>       g_poll_stop{false};
static cpu_usage_tracker       g_cpu_tracker;
#if defined(__linux__) || defined(__APPLE__)
static nvml_sampler            g_nvml;
#endif

static int poll_interval_ms() {
    const char * v = std::getenv("DIST_PERF_GPU_POLL_MS");
    if (v == nullptr || v[0] == '\0') {
        return 100;
    }
    const int ms = std::atoi(v);
    return ms > 0 ? ms : 100;
}

static void emit_gpu_sample(const gpu_sample_reading & reading) {
    char attrs[160];
    std::snprintf(attrs, sizeof(attrs),
            "{\"cpu_busy_pct\":%.2f,\"util_valid\":%s}",
            reading.cpu_busy_pct,
            reading.util_valid ? "true" : "false");
    const float util = reading.util_valid ? reading.util_pct : reading.cpu_busy_pct;
    perf_emit_gpu_sample(reading.backend, util, reading.mem_used_mb, attrs);
}

static gpu_sample_reading take_sample() {
    gpu_sample_reading reading {};
    reading.cpu_busy_pct = g_cpu_tracker.sample_busy_pct();

#if defined(__linux__) || defined(__APPLE__)
    if (g_nvml.sample(reading)) {
        return reading;
    }
#endif

#if defined(__APPLE__)
    reading.backend = "metal";
#else
    reading.backend = "cpu";
#endif
    reading.util_pct = reading.cpu_busy_pct;
    reading.util_valid = false;
    return reading;
}

static void poll_loop() {
    while (!g_poll_stop.load(std::memory_order_relaxed)) {
        if (perf_trace_has_active_context()) {
            emit_gpu_sample(take_sample());
        }
        const int ms = poll_interval_ms();
        for (int i = 0; i < ms && !g_poll_stop.load(std::memory_order_relaxed); ++i) {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
    }
}

static void start_thread_if_needed() {
    std::lock_guard<std::mutex> lock(g_poll_mu);
    if (g_poll_thread.joinable()) {
        return;
    }
    g_poll_stop.store(false, std::memory_order_relaxed);
    g_poll_thread = std::thread(poll_loop);
}

static void stop_thread_if_running() {
    std::thread worker;
    {
        std::lock_guard<std::mutex> lock(g_poll_mu);
        if (!g_poll_thread.joinable()) {
            return;
        }
        g_poll_stop.store(true, std::memory_order_relaxed);
        worker = std::move(g_poll_thread);
    }
    if (worker.joinable()) {
        worker.join();
    }
    g_poll_stop.store(false, std::memory_order_relaxed);
}

} // namespace

void perf_gpu_poll_start() {
    if (!perf_trace_enabled()) {
        return;
    }
    const int prev = g_poll_refs.fetch_add(1, std::memory_order_relaxed);
    if (prev == 0) {
        start_thread_if_needed();
    }
}

void perf_gpu_poll_stop() {
    int prev = g_poll_refs.fetch_sub(1, std::memory_order_relaxed);
    if (prev <= 1) {
        g_poll_refs.store(0, std::memory_order_relaxed);
        stop_thread_if_running();
        return;
    }
}
