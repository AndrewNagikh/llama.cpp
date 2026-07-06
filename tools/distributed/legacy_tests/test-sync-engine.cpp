#include "node_agent/synchronization/synchronization_engine.h"
#include "orchestrator/install_planner/install_planner.h"
#include "test_sync_common.h"

#include <cstdio>
#include <fstream>

int main() {
    const auto src = std::filesystem::temp_directory_path() / "dist-sync-engine.bin";
    {
        std::ofstream out(src, std::ios::binary);
        std::vector<uint8_t> data(8192, 0x7A);
        out.write(reinterpret_cast<const char *>(data.data()), data.size());
    }

    const model_manifest manifest = make_manifest_with_layer_ranges(2, 1024, 0);
    auto store = make_temp_layer_store("sync-engine");
    synchronization_engine engine;

    const std::string url = "file://" + src.string();
    install_plan plan;
    plan.model_id = "sync-engine";
    plan.operations.push_back(make_download_op_for_layer(0, "node-a", url, 0, 1024));
    plan.operations.push_back(make_download_op_for_layer(1, "node-a", url, 1024, 1024));

    const std::string job_id = engine.start_job(plan.model_id, plan.operations, store, &manifest);
    if (!engine.wait_job(job_id, 15000)) {
        fprintf(stderr, "test-sync-engine: job failed\n");
        return 1;
    }

    if (store.list_layers().size() != 2) {
        fprintf(stderr, "test-sync-engine: expected 2 layers\n");
        return 1;
    }

    const auto restored = store.load_manifest();
    if (!restored.has_value() || restored->layers.size() != 2) {
        fprintf(stderr, "test-sync-engine: manifest not persisted\n");
        return 1;
    }

    printf("test-sync-engine: OK\n");
    return 0;
}
