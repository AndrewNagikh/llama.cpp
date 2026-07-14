#pragma once

#include "../transport/split_tcp_wire.h"

#include <condition_variable>
#include <cstdint>
#include <deque>
#include <mutex>
#include <vector>

struct wave_work_item {
    split_gen_a_req        req{};
    std::vector<int32_t>   tokens;
    std::vector<float>     hidden;
    int32_t                hidden_n_embd = 0;
    int32_t                wave_id       = -1;
};

struct hidden_wave_work_item {
    split_tcp_hidden_msg msg;
    int32_t              wave_id    = -1;
    int32_t              debug_step = 0;
};

class wave_inbound_queue {
public:
    explicit wave_inbound_queue(int32_t max_depth = 2);

    int32_t max_depth() const;
    int32_t depth() const;
    int32_t observable_depth(bool processor_active) const;

    // Blocks when full (backpressure). Returns false if the queue is closed.
    bool push(wave_work_item item);

    bool try_pop(wave_work_item & item);
    // Blocks until an item is available or the queue is closed. Returns
    // false only when closed AND drained -- pending items are still
    // delivered after close().
    bool pop(wave_work_item & item);

    // Wakes every blocked pop()/push(). The producer (receiver thread) must
    // call this when it stops, otherwise a consumer parked in pop() sleeps
    // forever -- there is no other wakeup source.
    void close();

private:
    const int32_t          max_depth_;
    mutable std::mutex     mu_;
    std::condition_variable not_full_;
    std::condition_variable not_empty_;
    std::deque<wave_work_item> items_;
    bool                   closed_ = false;
};

class hidden_inbound_queue {
public:
    explicit hidden_inbound_queue(int32_t max_depth = 2);

    int32_t max_depth() const;
    int32_t depth() const;
    int32_t observable_depth(bool processor_active) const;

    bool push(hidden_wave_work_item item);
    bool try_pop(hidden_wave_work_item & item);
    // See wave_inbound_queue::pop.
    bool pop(hidden_wave_work_item & item);
    // See wave_inbound_queue::close.
    void close();

private:
    const int32_t          max_depth_;
    mutable std::mutex     mu_;
    std::condition_variable not_full_;
    std::condition_variable not_empty_;
    std::deque<hidden_wave_work_item> items_;
    bool                   closed_ = false;
};

int32_t runtime_entry_queue_max_depth();
bool    runtime_entry_queue_enabled();
int32_t runtime_stage_queue_max_depth();
bool    runtime_stage_queue_enabled();
bool    runtime_client_pipeline_enabled();
