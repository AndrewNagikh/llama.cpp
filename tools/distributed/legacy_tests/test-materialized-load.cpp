#include "test_sync_common.h"
#include "test_verify_common.h"

#include "ggml-backend.h"
#include "llama.h"

#include <cstdio>
#include <filesystem>

int main() {
    const std::string path = manifest_test_model_path();
    if (path.empty() || !std::filesystem::exists(path)) {
        fprintf(stderr, "test-materialized-load: GGUF model not found (set MODEL env)\n");
        return 77;
    }

    model_manifest manifest;
    auto store = make_temp_layer_store("materialized-load");
    if (!verify_setup_store_from_model(path, store, manifest)) {
        fprintf(stderr, "test-materialized-load: failed to populate layer store\n");
        return 1;
    }

    const std::string work = verify_test_work_dir("materialized-load");
    const std::string mat  = verify_materialize_from_store(store, manifest, work);
    if (mat.empty()) {
        fprintf(stderr, "test-materialized-load: materialize failed\n");
        return 1;
    }

    ggml_backend_load_all();
    llama_model_params mparams = llama_model_default_params();
    mparams.n_gpu_layers = 0;
    llama_model * model = llama_model_load_from_file(mat.c_str(), mparams);
    if (!model) {
        fprintf(stderr, "test-materialized-load: llama failed to load materialized GGUF\n");
        return 1;
    }

    const int32_t n_layer = llama_model_n_layer(model);
    llama_model_free(model);

    if (n_layer <= 0) {
        fprintf(stderr, "test-materialized-load: invalid layer count\n");
        return 1;
    }

    printf("test-materialized-load: OK n_layer=%d\n", n_layer);
    return 0;
}
