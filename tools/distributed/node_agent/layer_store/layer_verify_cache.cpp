#include "layer_verify_cache.h"

#include <chrono>
#include <map>
#include <mutex>

namespace {

struct cache_value {
    bool    ready       = false;
    int64_t verified_at = 0;
};

std::mutex                         g_mu;
std::map<std::string, cache_value> g_cache;

int64_t now_unix() {
    return std::chrono::duration_cast<std::chrono::seconds>(
            std::chrono::system_clock::now().time_since_epoch())
            .count();
}

std::string make_key(const std::string & model_id, const std::string & blob_key) {
    return model_id + "\x1f" + blob_key;
}

} // namespace

bool layer_verify_cache_get(
        const std::string & model_id,
        const std::string & blob_key,
        const int64_t ttl_sec,
        bool & out_ready) {
    std::lock_guard<std::mutex> lock(g_mu);
    const auto it = g_cache.find(make_key(model_id, blob_key));
    if (it == g_cache.end()) {
        return false;
    }
    if (now_unix() - it->second.verified_at > ttl_sec) {
        return false;
    }
    out_ready = it->second.ready;
    return true;
}

void layer_verify_cache_put(
        const std::string & model_id,
        const std::string & blob_key,
        const bool ready) {
    std::lock_guard<std::mutex> lock(g_mu);
    g_cache[make_key(model_id, blob_key)] = { ready, now_unix() };
}
