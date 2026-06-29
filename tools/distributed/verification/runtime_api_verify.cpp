#include "runtime_api_verify.h"

#include "runtime_debug/hidden_transport.h"

#include "ggml-backend.h"
#include "llama-distributed.h"

#include <vector>

runtime_api_verify_result verify_runtime_hidden_api(
        const std::string & model_path,
        const int32_t n_tokens) {
    runtime_api_verify_result result{};

    ggml_backend_load_all();
    llama_model * model = llama_model_load_from_file(model_path.c_str(), llama_model_default_params());
    if (!model) {
        result.message = "model load failed";
        return result;
    }

    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx   = 512;
    cparams.n_batch = 512;
    cparams.no_perf = true;
    llama_context * ctx = llama_init_from_model(model, cparams);
    if (!ctx) {
        llama_model_free(model);
        result.message = "context init failed";
        return result;
    }

    const int32_t n_embd = llama_model_n_embd(model);
    std::vector<float> data((size_t) n_tokens * n_embd);
    for (size_t i = 0; i < data.size(); ++i) {
        data[i] = static_cast<float>((int) i % 17) * 0.01f - 0.08f;
    }

    const hidden_state_api_check chk = verify_hidden_state_roundtrip(
            ctx, data.data(), n_tokens, n_embd);
    result.ok           = chk.ok;
    result.diff_offset  = chk.diff_offset;
    result.message      = chk.message;

    llama_free(ctx);
    llama_model_free(model);
    return result;
}
