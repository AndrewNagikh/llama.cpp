#include "test_sync_common.h"
#include "test_verify_common.h"
#include "verification/parity_verify.h"

#include <cstdio>
#include <filesystem>

int main() {
    const std::string path = manifest_test_model_path();
    if (path.empty() || !std::filesystem::exists(path)) {
        fprintf(stderr, "test-sampling-parity: GGUF model not found (set MODEL env)\n");
        return 77;
    }

    model_manifest manifest;
    auto store = make_temp_layer_store("sampling-parity");
    if (!verify_setup_store_from_model(path, store, manifest)) {
        fprintf(stderr, "test-sampling-parity: failed to populate layer store\n");
        return 1;
    }

    const std::string work = verify_test_work_dir("sampling-parity");
    const std::string mat  = verify_materialize_from_store(store, manifest, work);
    if (mat.empty()) {
        fprintf(stderr, "test-sampling-parity: materialize failed\n");
        return 1;
    }

    const auto parity = verify_inference_parity(path, mat, "The capital of France is", 32);
    if (parity.sampling.status != verify_status::ok) {
        fprintf(stderr, "test-sampling-parity: %s\n", parity.sampling.message.c_str());
        fprintf(stderr, "  original:     %zu tokens\n", parity.tokens_original.size());
        fprintf(stderr, "  materialized: %zu tokens\n", parity.tokens_materialized.size());
        return 1;
    }

    const size_t n8  = std::min<size_t>(8, parity.tokens_original.size());
    const size_t n32 = std::min<size_t>(32, parity.tokens_original.size());
    printf("test-sampling-parity: OK first=%d tokens8=%zu tokens32=%zu\n",
            parity.tokens_original.empty() ? -1 : parity.tokens_original[0],
            n8, n32);
    return 0;
}
