#include "test_sync_common.h"
#include "test_verify_common.h"
#include "verification/parity_verify.h"

#include <cstdio>
#include <filesystem>

int main() {
    const std::string path = manifest_test_model_path();
    if (path.empty() || !std::filesystem::exists(path)) {
        fprintf(stderr, "test-logit-parity: GGUF model not found (set MODEL env)\n");
        return 77;
    }

    model_manifest manifest;
    auto store = make_temp_layer_store("logit-parity");
    if (!verify_setup_store_from_model(path, store, manifest)) {
        fprintf(stderr, "test-logit-parity: failed to populate layer store\n");
        return 1;
    }

    const std::string work = verify_test_work_dir("logit-parity");
    const std::string mat  = verify_materialize_from_store(store, manifest, work);
    if (mat.empty()) {
        fprintf(stderr, "test-logit-parity: materialize failed\n");
        return 1;
    }

    const auto parity = verify_inference_parity(path, mat);
    if (parity.logits.status != verify_status::ok) {
        fprintf(stderr, "test-logit-parity: %s\n", parity.logits.message.c_str());
        return 1;
    }

    printf("test-logit-parity: OK\n");
    return 0;
}
