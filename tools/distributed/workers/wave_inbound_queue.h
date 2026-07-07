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

    // Blocks when full (backpressure).
    bool push(wave_work_item item);

    bool try_pop(wave_work_item & item);
    bool pop(wave_work_item & item);

private:
    const int32_t          max_depth_;
    mutable std::mutex     mu_;
    std::condition_variable not_full_;
    std::condition_variable not_empty_;
    std::deque<wave_work_item> items_;
};

class hidden_inbound_queue {
public:
    explicit hidden_inbound_queue(int32_t max_depth = 2);

    int32_t max_depth() const;
    int32_t depth() const;
    int32_t observable_depth(bool processor_active) const;

    bool push(hidden_wave_work_item item);
    bool try_pop(hidden_wave_work_item & item);
    bool pop(hidden_wave_work_item & item);

private:
    const int32_t          max_depth_;
    mutable std::mutex     mu_;
    std::condition_variable not_full_;
    std::condition_variable not_empty_;
    std::deque<hidden_wave_work_item> items_;
};

int32_t runtime_entry_queue_max_depth();
bool    runtime_entry_queue_enabled();
int32_t runtime_stage_queue_max_depth();
bool    runtime_stage_queue_enabled();
bool    runtime_client_pipeline_enabled();
