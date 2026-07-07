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

static void test_depth_and_backpressure() {
    wave_inbound_queue q(2);
    check(q.max_depth() == 2, "max_depth");
    check(q.push(wave_work_item{}), "push 1");
    check(q.push(wave_work_item{}), "push 2");
    check(q.depth() == 2, "depth 2");
    check(q.observable_depth(true) == 3, "observable depth with processor");

    std::atomic<bool> pushed{false};
    std::thread blocked([&]() {
        wave_work_item item;
        item.wave_id = 99;
        q.push(std::move(item));
        pushed = true;
    });

    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    check(!pushed.load(), "third push blocks when full");

    wave_work_item popped;
    check(q.try_pop(popped), "pop frees slot");
    blocked.join();
    check(pushed.load(), "blocked push completes after pop");
    check(q.depth() == 2, "depth restored to 2");
}

int main() {
    test_depth_and_backpressure();
    if (failures > 0) {
        fprintf(stderr, "%d test(s) failed\n", failures);
        return 1;
    }
    fprintf(stderr, "test-wave-inbound-queue: ok\n");
    return 0;
}
