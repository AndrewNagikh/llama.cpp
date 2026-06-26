#include "node_agent/synchronization/synchronization_engine.h"
#include "test_sync_common.h"

#include <cstdio>
#include <fstream>

int main() {
    const auto src = std::filesystem::temp_directory_path() / "dist-partial-install.bin";
    {
        std::ofstream out(src, std::ios::binary);
        std::vector<uint8_t> data(4096, 0x33);
        out.write(reinterpret_cast<const char *>(data.data()), data.size());
    }

    auto store = make_temp_layer_store("partial");
    synchronization_engine engine;
    const std::string url = "file://" + src.string();

    const install_operation first = make_download_op_for_layer(0, "node-a", url, 0, 512);
    const std::string job1 = engine.start_job("partial", { first }, store, nullptr);
    if (!engine.wait_job(job1, 10000) || !store.has_layer(0)) {
        fprintf(stderr, "test-partial-install: first layer failed\n");
        return 1;
    }

    const install_operation second = make_download_op_for_layer(1, "node-a", url, 512, 512);
    const std::string job2 = engine.start_job("partial", { second }, store, nullptr);
    if (!engine.wait_job(job2, 10000) || store.list_layers().size() != 2) {
        fprintf(stderr, "test-partial-install: resume install failed\n");
        return 1;
    }

    printf("test-partial-install: OK\n");
    return 0;
}
