#include "wave_inbound_queue.h"

#include "../transport/runtime_protocol.h"

#include <algorithm>
#include <cstdlib>
#include <cstring>

namespace {

bool env_truthy(const char * name) {
    const char * v = std::getenv(name);
    if (v == nullptr || v[0] == '\0') {
        return false;
    }
    return std::strcmp(v, "0") != 0 &&
           std::strcmp(v, "false") != 0 &&
           std::strcmp(v, "FALSE") != 0;
}

} // namespace

int32_t runtime_entry_queue_max_depth() {
    const char * v = std::getenv("DIST_RUNTIME_ENTRY_QUEUE_DEPTH");
    if (v == nullptr || v[0] == '\0') {
        return 2;
    }
    const int d = std::atoi(v);
    return d < 1 ? 1 : (d > 32 ? 32 : d);
}

bool runtime_entry_queue_enabled() {
    const char * v = std::getenv("DIST_RUNTIME_ENTRY_QUEUE");
    if (v != nullptr && v[0] != '\0') {
        return env_truthy("DIST_RUNTIME_ENTRY_QUEUE");
    }
    return runtime_protocol_v2_enabled();
}

int32_t runtime_stage_queue_max_depth() {
    const char * v = std::getenv("DIST_RUNTIME_STAGE_QUEUE_DEPTH");
    if (v == nullptr || v[0] == '\0') {
        return runtime_entry_queue_max_depth();
    }
    const int d = std::atoi(v);
    return d < 1 ? 1 : (d > 32 ? 32 : d);
}

bool runtime_stage_queue_enabled() {
    const char * v = std::getenv("DIST_RUNTIME_STAGE_QUEUE");
    if (v != nullptr && v[0] != '\0') {
        return env_truthy("DIST_RUNTIME_STAGE_QUEUE");
    }
    return runtime_protocol_v2_enabled();
}

bool runtime_client_pipeline_enabled() {
    const char * v = std::getenv("DIST_RUNTIME_CLIENT_PIPELINE");
    if (v != nullptr && v[0] != '\0') {
        return env_truthy("DIST_RUNTIME_CLIENT_PIPELINE");
    }
    return runtime_entry_queue_enabled() && runtime_stage_queue_enabled();
}

wave_inbound_queue::wave_inbound_queue(const int32_t max_depth)
    : max_depth_(std::max(1, max_depth)) {}

int32_t wave_inbound_queue::max_depth() const {
    return max_depth_;
}

int32_t wave_inbound_queue::depth() const {
    std::lock_guard<std::mutex> lock(mu_);
    return (int32_t) items_.size();
}

int32_t wave_inbound_queue::observable_depth(const bool processor_active) const {
    return depth() + (processor_active ? 1 : 0);
}

bool wave_inbound_queue::push(wave_work_item item) {
    std::unique_lock<std::mutex> lock(mu_);
    not_full_.wait(lock, [&] { return (int32_t) items_.size() < max_depth_; });
    items_.push_back(std::move(item));
    not_empty_.notify_one();
    return true;
}

bool wave_inbound_queue::try_pop(wave_work_item & item) {
    std::lock_guard<std::mutex> lock(mu_);
    if (items_.empty()) {
        return false;
    }
    item = std::move(items_.front());
    items_.pop_front();
    not_full_.notify_one();
    return true;
}

bool wave_inbound_queue::pop(wave_work_item & item) {
    std::unique_lock<std::mutex> lock(mu_);
    not_empty_.wait(lock, [&] { return !items_.empty(); });
    item = std::move(items_.front());
    items_.pop_front();
    not_full_.notify_one();
    return true;
}

hidden_inbound_queue::hidden_inbound_queue(const int32_t max_depth)
    : max_depth_(std::max(1, max_depth)) {}

int32_t hidden_inbound_queue::max_depth() const {
    return max_depth_;
}

int32_t hidden_inbound_queue::depth() const {
    std::lock_guard<std::mutex> lock(mu_);
    return (int32_t) items_.size();
}

int32_t hidden_inbound_queue::observable_depth(const bool processor_active) const {
    return depth() + (processor_active ? 1 : 0);
}

bool hidden_inbound_queue::push(hidden_wave_work_item item) {
    std::unique_lock<std::mutex> lock(mu_);
    not_full_.wait(lock, [&] { return (int32_t) items_.size() < max_depth_; });
    items_.push_back(std::move(item));
    not_empty_.notify_one();
    return true;
}

bool hidden_inbound_queue::try_pop(hidden_wave_work_item & item) {
    std::lock_guard<std::mutex> lock(mu_);
    if (items_.empty()) {
        return false;
    }
    item = std::move(items_.front());
    items_.pop_front();
    not_full_.notify_one();
    return true;
}

bool hidden_inbound_queue::pop(hidden_wave_work_item & item) {
    std::unique_lock<std::mutex> lock(mu_);
    not_empty_.wait(lock, [&] { return !items_.empty(); });
    item = std::move(items_.front());
    items_.pop_front();
    not_full_.notify_one();
    return true;
}
