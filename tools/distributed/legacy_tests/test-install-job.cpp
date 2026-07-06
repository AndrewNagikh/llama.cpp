#include "node_agent/synchronization/synchronization_engine.h"
#include "test_sync_common.h"

#include <cstdio>
#include <fstream>

int main() {
    auto store = make_temp_layer_store("install-job");
    synchronization_engine engine;

    const std::string file_path = []() {
        const auto tmp = std::filesystem::temp_directory_path() / "dist-install-job.bin";
        std::ofstream out(tmp, std::ios::binary);
        std::vector<uint8_t> data(2048, 0x11);
        out.write(reinterpret_cast<const char *>(data.data()), data.size());
        return tmp.string();
    }();

    const std::string url = "file://" + file_path;
    const install_operation op = make_download_op_for_layer(1, "node-a", url, 256, 512);

    const std::string job_id = engine.start_job("install-job", { op }, store, nullptr);
    if (job_id.empty()) {
        fprintf(stderr, "test-install-job: empty job id\n");
        return 1;
    }

    if (!engine.wait_job(job_id, 10000)) {
        fprintf(stderr, "test-install-job: job did not complete\n");
        return 1;
    }

    const auto job = engine.get_job(job_id);
    if (!job.has_value() || job->state != sync_job_state::completed) {
        fprintf(stderr, "test-install-job: unexpected job state\n");
        return 1;
    }
    if (job->operations.size() != 1 ||
            job->operations[0].state != sync_operation_state::ready) {
        fprintf(stderr, "test-install-job: operation not READY\n");
        return 1;
    }

    printf("test-install-job: OK\n");
    return 0;
}
