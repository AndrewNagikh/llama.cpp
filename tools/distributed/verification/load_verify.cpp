#include "load_verify.h"

#include "gguf_inspector.h"
#include "verification_common.h"

#include "ggml-backend.h"
#include "llama.h"

load_verify_result load_verify_gguf(const std::string & gguf_path) {
    load_verify_result result;
    result.summary.name = "stage_8_runtime_load";

    const auto info = gguf_inspect_file(gguf_path);
    if (info.contains("tensor_count")) {
        result.tensor_count = info["tensor_count"].get<int64_t>();
    }
    if (info.contains("kv_count")) {
        result.kv_size = info["kv_count"].get<int32_t>();
    }

    ggml_backend_load_all();
    llama_model_params mparams = llama_model_default_params();
    mparams.n_gpu_layers       = 0;
    llama_model * model = llama_model_load_from_file(gguf_path.c_str(), mparams);
    if (!model) {
        result.summary.status  = verify_status::fail;
        result.summary.message = "llama_model_load failed";
        return result;
    }

    result.n_layer      = llama_model_n_layer(model);

    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx   = 512;
    cparams.n_batch = 512;
    cparams.no_perf = true;
    llama_context * ctx = llama_init_from_model(model, cparams);
    if (!ctx) {
        llama_model_free(model);
        result.summary.status  = verify_status::fail;
        result.summary.message = "llama_init_from_model failed";
        return result;
    }

    result.metadata_sha256 = sha256_file_hex(gguf_path);

    llama_free(ctx);
    llama_model_free(model);

    result.summary.status  = verify_status::ok;
    result.summary.message = "model loaded successfully";
    result.summary.details = {
        { "layer_count", result.n_layer },
        { "offloaded_layers", result.n_gpu_layers },
        { "tensor_count", result.tensor_count },
        { "kv_count", result.kv_size },
        { "metadata_sha256", result.metadata_sha256 },
    };
    return result;
}
