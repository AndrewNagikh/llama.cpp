#include "runtime/llama_layer_store_model_load.h"
#include "test_sync_common.h"
#include "test_verify_common.h"

#include "ggml-backend.h"

#include <cstdio>
#include <filesystem>

int main() {
    const std::string path = manifest_test_model_path();
    if (path.empty() || !std::filesystem::exists(path)) {
        fprintf(stderr, "test-runtime-layer-store-model-load: GGUF model not found (set MODEL env)\n");
        return 77;
    }

    model_manifest manifest;
    auto store = make_temp_layer_store("layer-store-model-load");
    if (!verify_setup_store_from_model(path, store, manifest)) {
        fprintf(stderr, "test-runtime-layer-store-model-load: failed to populate layer store\n");
        return 1;
    }

    const std::string work = verify_test_work_dir("layer-store-model-load");
    const std::string mat  = verify_materialize_from_store(store, manifest, work);
    if (mat.empty()) {
        fprintf(stderr, "test-runtime-layer-store-model-load: materialize failed\n");
        return 1;
    }

    ggml_backend_load_all();
    llama_model_params params = llama_model_default_params();
    params.n_gpu_layers = 0;

    llama_model * file_model = llama_model_load_from_file(mat.c_str(), params);
    if (!file_model) {
        fprintf(stderr, "test-runtime-layer-store-model-load: file load failed\n");
        return 1;
    }

    runtime_layer_store_model_load_request req{};
    req.role        = worker_role::full;
    req.layer_start = 0;
    req.layer_end   = static_cast<int32_t>(manifest.n_layer);
    req.params      = params;

    std::string err;
    llama_model * provider_model = runtime_load_model_from_layer_store(
            store, manifest, req, err);
    if (!provider_model) {
        llama_model_free(file_model);
        fprintf(stderr, "test-runtime-layer-store-model-load: provider load failed: %s\n", err.c_str());
        return 1;
    }

    const int32_t file_layers = llama_model_n_layer(file_model);
    const int32_t prov_layers = llama_model_n_layer(provider_model);
    const int32_t file_embd   = llama_model_n_embd(file_model);
    const int32_t prov_embd   = llama_model_n_embd(provider_model);

    runtime_free_model(provider_model);
    llama_model_free(file_model);

    if (file_layers != prov_layers || file_embd != prov_embd) {
        fprintf(stderr,
                "test-runtime-layer-store-model-load: mismatch layers %d/%d embd %d/%d\n",
                file_layers,
                prov_layers,
                file_embd,
                prov_embd);
        return 1;
    }

    printf("test-runtime-layer-store-model-load: OK n_layer=%d n_embd=%d\n", file_layers, file_embd);
    return 0;
}
