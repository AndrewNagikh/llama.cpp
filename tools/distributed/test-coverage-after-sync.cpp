#include "node_agent/layer_store/layer_store.h"
#include "node_agent/synchronization/synchronization_engine.h"
#include "orchestrator/coverage/coverage.h"
#include "test_coverage_common.h"
#include "test_sync_common.h"

#include <cstdio>
#include <fstream>

int main() {
    const auto src = std::filesystem::temp_directory_path() / "dist-coverage-sync.bin";
    {
        std::ofstream out(src, std::ios::binary);
        std::vector<uint8_t> data(4096, 0x66);
        out.write(reinterpret_cast<const char *>(data.data()), data.size());
    }

    auto store = make_temp_layer_store("coverage-sync");
    synchronization_engine engine;
    const std::string url = "file://" + src.string();

    const desired_model_layout desired = make_desired_layout("coverage-sync", {
        { "node-a", 0 }, { "node-a", 1 },
    });

    std::vector<install_operation> ops;
    ops.push_back(make_download_op_for_layer(0, "node-a", url, 0, 1024));
    ops.push_back(make_download_op_for_layer(1, "node-a", url, 1024, 1024));

    const std::string job_id = engine.start_job("coverage-sync", ops, store, nullptr);
    if (!engine.wait_job(job_id, 15000)) {
        fprintf(stderr, "test-coverage-after-sync: sync failed\n");
        return 1;
    }

    actual_model_layout actual;
    actual.model_id = "coverage-sync";

    for (const layer_blob & blob : store.list_layers()) {
        installed_layer layer = make_ready_layer(blob.layer_index, "node-a", blob.checksum);
        layer.size_bytes = blob.size_bytes;
        if (!store.verify_layer(blob.layer_index, blob.checksum)) {
            layer.state = install_state::corrupted;
        }
        actual.layers.push_back(layer);
    }

    const coverage_report report = compute_coverage(desired, actual);
    if (report.state != coverage_state::ready) {
        fprintf(stderr, "test-coverage-after-sync: expected READY got %s\n",
                coverage_state_to_string(report.state).c_str());
        return 1;
    }

    printf("test-coverage-after-sync: OK\n");
    return 0;
}
