#include "split_gen_model_load.h"

#include "runtime/llama_layer_store_model_load.h"
#include "runtime/runtime_config.h"

#include "architecture/semantic_runtime_descriptor.h"
#include "architecture/worker_requirement.h"
#include "node_agent/layer_store/layer_store.h"

#include "ggml-backend.h"

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <vector>

namespace {

bool env_flag(const char * name) {
    const char * v = std::getenv(name);
    return v != nullptr && v[0] == '1' && v[1] == '\0';
}

bool all_nodes_blob_required_for_role(const semantic_blob & blob, const worker_role role) {
    if (blob.deploy != blob_deploy_target::all_nodes) {
        return false;
    }
    if (blob.id == "embedding" || blob.role == tensor_semantic_role::embedding) {
        return role == worker_role::embedding ||
               role == worker_role::full;
    }
    return true;
}

std::vector<std::string> required_blobs_for_worker(
        const semantic_runtime_descriptor & rt,
        const worker_role role,
        const int32_t layer_start,
        const int32_t layer_end) {
    const worker_materialize_plan plan =
            worker_materialize_plan_for_role(rt, role, layer_start, layer_end);
    std::vector<std::string> required = plan.required_blobs;
    for (const semantic_blob & blob : rt.blobs) {
        if (all_nodes_blob_required_for_role(blob, role)) {
            required.push_back(blob.id);
        }
    }
    for (int32_t layer = layer_start; layer < layer_end; ++layer) {
        required.push_back(layer_blob_id(layer));
    }
    std::sort(required.begin(), required.end());
    required.erase(std::unique(required.begin(), required.end()), required.end());
    return required;
}

worker_role worker_role_from_env_string(const std::string & name) {
    if (name == "PIPELINE_STAGE" || name == "pipeline_stage") {
        return worker_role::pipeline_stage;
    }
    if (name == "EMBEDDING" || name == "embedding") {
        return worker_role::embedding;
    }
    if (name == "OUTPUT_HEAD" || name == "output_head") {
        return worker_role::output_head;
    }
    if (name == "TOKENIZER" || name == "tokenizer") {
        return worker_role::tokenizer;
    }
    return worker_role::pipeline_stage;
}

std::string models_dir_from_env() {
    if (const char * v = std::getenv("DIST_LAYER_STORE_ROOT")) {
        if (v[0] != '\0') {
            return v;
        }
    }
    if (const char * v = std::getenv("MODELS_DIR")) {
        if (v[0] != '\0') {
            return v;
        }
    }
    return {};
}

bool layer_store_load_enabled() {
    return runtime_layer_first_enabled() || env_flag("DIST_RUNTIME_LAYER_FIRST");
}

} // namespace

bool split_gen_model_uses_layer_store() {
    if (!layer_store_load_enabled()) {
        return false;
    }
    const char * model_id = std::getenv("DIST_MODEL_ID");
    return model_id != nullptr && model_id[0] != '\0';
}

llama_model * split_gen_load_model(const char * model_path, std::string & err) {
    err.clear();
    ggml_backend_load_all();

    if (split_gen_model_uses_layer_store()) {
        const char * model_id = std::getenv("DIST_MODEL_ID");
        const std::string models_dir = models_dir_from_env();
        if (models_dir.empty()) {
            err = "DIST_LAYER_STORE_ROOT or MODELS_DIR required for layer-first worker load";
            return nullptr;
        }

        layer_store store(models_dir, model_id);
        const auto manifest = store.load_manifest();
        if (!manifest.has_value()) {
            err = "layer store manifest missing for worker";
            return nullptr;
        }

        worker_role role = worker_role::pipeline_stage;
        if (const char * role_env = std::getenv("DIST_WORKER_ROLE")) {
            role = worker_role_from_env_string(role_env);
        }

        int32_t layer_start = 0;
        int32_t layer_end   = static_cast<int32_t>(manifest->n_layer);
        if (const char * v = std::getenv("DIST_WORKER_LAYER_START")) {
            layer_start = std::atoi(v);
        }
        if (const char * v = std::getenv("DIST_WORKER_LAYER_END")) {
            layer_end = std::atoi(v);
        }

        runtime_layer_store_model_load_request req{};
        req.role        = role;
        req.layer_start = layer_start;
        req.layer_end   = layer_end;
        req.params      = llama_model_default_params();
        if (const char * v = std::getenv("DIST_WORKER_VOCAB_ONLY")) {
            if (v[0] == '1' && v[1] == '\0') {
                req.params.vocab_only = true;
            }
        }

        fprintf(stderr,
                "split_gen: loading model %s from layer store role=%s layers=[%d,%d)\n",
                model_id,
                worker_role_to_string(role).c_str(),
                layer_start,
                layer_end);
        return runtime_load_model_from_layer_store(store, *manifest, req, err);
    }

    if (model_path == nullptr || model_path[0] == '\0') {
        err = "model path required";
        return nullptr;
    }
    llama_model * model = llama_model_load_from_file(model_path, llama_model_default_params());
    if (!model) {
        err = "llama_model_load_from_file failed";
    }
    return model;
}
