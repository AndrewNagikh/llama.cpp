#include "node_agent/synchronization/executors/http_range/http_range_executor.h"
#include "test_sync_common.h"

#include <cstdio>
#include <fstream>
#include <vector>

int main() {
    const auto tmp = std::filesystem::temp_directory_path() / "dist-http-range-src.bin";
    std::vector<uint8_t> source(4096);
    for (size_t i = 0; i < source.size(); ++i) {
        source[i] = static_cast<uint8_t>(i & 0xFF);
    }
    {
        std::ofstream out(tmp, std::ios::binary);
        out.write(reinterpret_cast<const char *>(source.data()), source.size());
    }

    auto store = make_temp_layer_store("http-range");
    const std::string file_url = "file://" + tmp.string();
    const install_operation op = make_download_op_for_layer(0, "node-a", file_url, 512, 1024);

    http_range_download_executor executor;
    const executor_result result = executor.execute(op, store);
    if (!result.success) {
        fprintf(stderr, "test-http-range-executor: execute failed: %s\n", result.error.c_str());
        return 1;
    }

    if (!store.has_layer(0) || !store.verify_layer(0, op.download.checksum)) {
        fprintf(stderr, "test-http-range-executor: stored layer invalid\n");
        return 1;
    }

    std::vector<uint8_t> blob;
    store.load_layer(0, blob);
    if (blob.size() != 1024 || blob[0] != source[512]) {
        fprintf(stderr, "test-http-range-executor: blob content mismatch\n");
        return 1;
    }

    printf("test-http-range-executor: OK\n");
    return 0;
}
