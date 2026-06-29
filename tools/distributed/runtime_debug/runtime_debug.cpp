#include "runtime_debug.h"

#include <chrono>
#include <cstdlib>
#include <cstring>

using namespace std::chrono;

static dist_debug_config g_cfg{};
static bool g_cfg_loaded = false;

static bool env_truthy(const char * name) {
    const char * v = std::getenv(name);
    if (v == nullptr || v[0] == '\0') {
        return false;
    }
    return std::strcmp(v, "0") != 0 &&
           std::strcmp(v, "false") != 0 &&
           std::strcmp(v, "FALSE") != 0 &&
           std::strcmp(v, "no") != 0 &&
           std::strcmp(v, "NO") != 0;
}

bool dist_debug_env_truthy(const char * name) {
    return env_truthy(name);
}

dist_debug_config dist_debug_load_config() {
    if (g_cfg_loaded) {
        return g_cfg;
    }

    g_cfg.enabled = env_truthy("LLAMA_DISTRIBUTED_DEBUG") || env_truthy("LLAMA_DIST_DEBUG") ||
                    env_truthy("LLAMA_RUNTIME_STATE");

    g_cfg.skip_sampler  = env_truthy("LLAMA_DIST_SKIP_SAMPLER");
    g_cfg.dump_raw_bins = env_truthy("LLAMA_DIST_DUMP_RAW");

    if (const char * dir = std::getenv("LLAMA_DIST_TRACE_DIR")) {
        g_cfg.trace_dir = dir;
    } else if (const char * models = std::getenv("MODELS_DIR")) {
        g_cfg.trace_dir = std::string(models) + "/traces";
    } else {
        g_cfg.trace_dir = "/tmp/dist_trace";
    }

    if (const char * sid = std::getenv("LLAMA_DIST_SESSION_ID")) {
        g_cfg.session_id = sid;
    }
    if (const char * wid = std::getenv("LLAMA_DIST_WORKER_ID")) {
        g_cfg.worker_id = wid;
    }
    if (const char * nid = std::getenv("LLAMA_DIST_NODE_ID")) {
        g_cfg.node_id = nid;
    }

    g_cfg_loaded = true;
    return g_cfg;
}

bool dist_debug_enabled() {
    dist_debug_load_config();
    return g_cfg.enabled;
}

bool dist_debug_skip_sampler() {
    dist_debug_load_config();
    return g_cfg.enabled && g_cfg.skip_sampler;
}

const dist_debug_config & dist_debug_config_get() {
    dist_debug_load_config();
    return g_cfg;
}

void dist_debug_set_session(const std::string & session_id) {
    dist_debug_load_config();
    g_cfg.session_id = session_id;
}

void dist_debug_set_worker(const std::string & worker_id, const std::string & node_id) {
    dist_debug_load_config();
    g_cfg.worker_id = worker_id;
    g_cfg.node_id   = node_id;
}

uint64_t dist_debug_now_ms() {
    return static_cast<uint64_t>(
            duration_cast<milliseconds>(system_clock::now().time_since_epoch()).count());
}
