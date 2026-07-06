#include "node_agent/synchronization/synchronization_engine.h"
#include "test_sync_common.h"

#include <chrono>
#include <cstdio>
#include <fstream>
#include <thread>

int main() {
    auto store = make_temp_layer_store("job-progress");
    synchronization_engine engine;

    const std::string file_path = []() {
        const auto tmp = std::filesystem::temp_directory_path() / "dist-job-progress.bin";
        std::ofstream out(tmp, std::ios::binary);
        std::vector<uint8_t> data(1024, 0x22);
        out.write(reinterpret_cast<const char *>(data.data()), data.size());
        return tmp.string();
    }();

    const std::string url = "file://" + file_path;
    std::vector<install_operation> ops;
    ops.push_back(make_download_op_for_layer(0, "node-a", url, 0, 256));
    ops.push_back(make_download_op_for_layer(1, "node-a", url, 256, 256));

    const std::string job_id = engine.start_job("job-progress", ops, store, nullptr);

    bool saw_running = false;
    for (int i = 0; i < 200; ++i) {
        const auto job = engine.get_job(job_id);
        if (!job.has_value()) {
            break;
        }
        if (job->state == sync_job_state::running) {
            saw_running = true;
        }
        if (job->state == sync_job_state::completed) {
            break;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
    }

    if (!saw_running) {
        fprintf(stderr, "test-job-progress: never saw RUNNING state\n");
        return 1;
    }

    if (!engine.wait_job(job_id, 10000)) {
        fprintf(stderr, "test-job-progress: job failed to complete\n");
        return 1;
    }

    const auto final_job = engine.get_job(job_id);
    if (!final_job.has_value() || final_job->ready_count() != 2) {
        fprintf(stderr, "test-job-progress: expected 2 ready operations\n");
        return 1;
    }

    printf("test-job-progress: OK\n");
    return 0;
}
