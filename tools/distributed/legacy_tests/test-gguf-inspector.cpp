#include "test_verify_common.h"
#include "verification/gguf_inspector.h"

#include <cstdio>
#include <filesystem>

int main() {
    const std::string path = manifest_test_model_path();
    if (path.empty() || !std::filesystem::exists(path)) {
        fprintf(stderr, "test-gguf-inspector: GGUF model not found (set MODEL env)\n");
        return 77;
    }

    const nlohmann::json j = gguf_inspect_file(path);
    if (j.contains("error")) {
        fprintf(stderr, "test-gguf-inspector: inspect failed: %s\n", j["error"].get<std::string>().c_str());
        return 1;
    }

    if (!j.contains("version") || j["version"].get<int>() < 2) {
        fprintf(stderr, "test-gguf-inspector: invalid version\n");
        return 1;
    }
    if (!j.contains("alignment") || j["alignment"].get<uint64_t>() == 0) {
        fprintf(stderr, "test-gguf-inspector: invalid alignment\n");
        return 1;
    }
    if (!j.contains("tensor_count") || j["tensor_count"].get<int64_t>() <= 0) {
        fprintf(stderr, "test-gguf-inspector: no tensors\n");
        return 1;
    }
    if (!j.contains("tensors") || !j["tensors"].is_array() || j["tensors"].empty()) {
        fprintf(stderr, "test-gguf-inspector: tensor list empty\n");
        return 1;
    }

    const auto & t0 = j["tensors"][0];
    if (!t0.contains("name") || !t0.contains("offset") || !t0.contains("size_bytes") ||
            !t0.contains("ggml_type")) {
        fprintf(stderr, "test-gguf-inspector: tensor entry missing fields\n");
        return 1;
    }

    printf("test-gguf-inspector: OK tensors=%lld\n", (long long) j["tensor_count"].get<int64_t>());
    return 0;
}
