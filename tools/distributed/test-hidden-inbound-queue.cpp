#include "workers/wave_inbound_queue.h"

#include <atomic>
#include <chrono>
#include <cstdio>
#include <thread>

static int failures = 0;

static void check(bool ok, const char * msg) {
    if (!ok) {
        fprintf(stderr, "FAIL: %s\n", msg);
        failures++;
    }
}

static void test_hidden_depth_and_backpressure() {
    hidden_inbound_queue q(2);
    check(q.max_depth() == 2, "max_depth");
    hidden_wave_work_item item;
    item.wave_id = 1;
    check(q.push(std::move(item)), "push 1");
    item.wave_id = 2;
    check(q.push(std::move(item)), "push 2");
    check(q.depth() == 2, "depth 2");
    check(q.observable_depth(true) == 3, "observable depth with processor");

    std::atomic<bool> pushed{false};
    std::thread blocked([&]() {
        hidden_wave_work_item blocked_item;
        blocked_item.wave_id = 99;
        q.push(std::move(blocked_item));
        pushed = true;
    });

    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    check(!pushed.load(), "third push blocks when full");

    hidden_wave_work_item popped;
    check(q.try_pop(popped), "pop frees slot");
    blocked.join();
    check(pushed.load(), "blocked push completes after pop");
    check(q.depth() == 2, "depth restored to 2");
}

int main() {
    test_hidden_depth_and_backpressure();
    if (failures > 0) {
        fprintf(stderr, "%d test(s) failed\n", failures);
        return 1;
    }
    fprintf(stderr, "test-hidden-inbound-queue: ok\n");
    return 0;
}
