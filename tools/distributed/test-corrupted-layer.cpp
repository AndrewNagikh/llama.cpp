#include "node_agent/synchronization/synchronization_engine.h"
#include "test_sync_common.h"

#include <cstdio>
#include <fstream>

int main() {
    const auto src = std::filesystem::temp_directory_path() / "dist-corrupted-layer.bin";
    {
        std::ofstream out(src, std::ios::binary);
        std::vector<uint8_t> data(1024, 0x44);
        out.write(reinterpret_cast<const char *>(data.data()), data.size());
    }

    auto store = make_temp_layer_store("corrupted");
    synchronization_engine engine;
    const std::string url = "file://" + src.string();

    install_operation bad = make_download_op_for_layer(0, "node-a", url, 0, 256);
    bad.download.checksum = "fnv1a:deadbeef";

    const std::string job_id = engine.start_job("corrupted", { bad }, store, nullptr);
    engine.wait_job(job_id, 10000);

    const auto job = engine.get_job(job_id);
    if (!job.has_value() || job->state != sync_job_state::failed) {
        fprintf(stderr, "test-corrupted-layer: expected FAILED job\n");
        return 1;
    }

  install_operation repair = make_download_op_for_layer(0, "node-a", url, 0, 256);
    const std::string repair_id = engine.start_job("corrupted", { repair }, store, nullptr);
    if (!engine.wait_job(repair_id, 10000) || !store.verify_layer(0, repair.download.checksum)) {
        fprintf(stderr, "test-corrupted-layer: repair failed\n");
        return 1;
    }

    printf("test-corrupted-layer: OK\n");
    return 0;
}
