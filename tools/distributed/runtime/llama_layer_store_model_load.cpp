#include "llama_layer_store_model_load.h"

#include "layer_store_tensor_provider.h"

#include "ggml-backend.h"
#include "gguf.h"

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <unordered_set>
#include <vector>

namespace {

struct runtime_llama_tensor_load_ctx {
    const layer_store_tensor_provider * provider = nullptr;
    const std::unordered_set<std::string> * required_tensors = nullptr;
    int32_t layer_start = 0;
    int32_t layer_end   = 0;
    std::string                         last_error;
};

bool runtime_tensor_layer_in_range(const char * tensor_name, const int32_t layer_start, const int32_t layer_end) {
    if (tensor_name == nullptr || std::strncmp(tensor_name, "blk.", 4) != 0) {
        return false;
    }
    char * end = nullptr;
    const long layer = std::strtol(tensor_name + 4, &end, 10);
    return end != nullptr && *end == '.' && layer >= layer_start && layer < layer_end;
}

bool runtime_llama_tensor_required(const char * tensor_name, void * userdata) {
    auto * ctx = static_cast<runtime_llama_tensor_load_ctx *>(userdata);
    if (ctx == nullptr || ctx->required_tensors == nullptr || tensor_name == nullptr) {
        return true;
    }
    if (runtime_tensor_layer_in_range(tensor_name, ctx->layer_start, ctx->layer_end)) {
        return true;
    }
    return ctx->required_tensors->find(tensor_name) != ctx->required_tensors->end();
}

void runtime_llama_set_tensor_data(struct ggml_tensor * tensor, void * userdata) {
    auto * ctx = static_cast<runtime_llama_tensor_load_ctx *>(userdata);
    if (ctx == nullptr || ctx->provider == nullptr || tensor == nullptr) {
        return;
    }

    const std::string name = ggml_get_name(tensor);
    const size_t      nbytes = ggml_nbytes(tensor);
    std::vector<uint8_t> data;
    if (ctx->provider->load_tensor(name, data)) {
        if (data.size() != nbytes) {
            ctx->last_error = "tensor size mismatch for " + name;
            return;
        }
        ggml_backend_tensor_set(tensor, data.data(), 0, nbytes);
        return;
    }

    if (ctx->provider->tensor_exists(name)) {
        ctx->last_error = "failed to read tensor " + name;
        return;
    }

    std::vector<uint8_t> zeros(nbytes, 0);
    ggml_backend_tensor_set(tensor, zeros.data(), 0, nbytes);
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

int32_t runtime_layer_count(const semantic_runtime_descriptor & rt) {
    int32_t n_layer = 0;
    for (const semantic_blob & blob : rt.blobs) {
        if (blob.role != tensor_semantic_role::transformer_layer) {
            continue;
        }
        const std::string prefix = "layer:";
        if (blob.id.rfind(prefix, 0) != 0) {
            continue;
        }
        char * end = nullptr;
        const long layer = std::strtol(blob.id.c_str() + prefix.size(), &end, 10);
        if (end != nullptr && *end == '\0' && layer >= 0) {
            n_layer = std::max(n_layer, (int32_t) layer + 1);
        }
    }
    return n_layer;
}

bool final_stage_output_required(
        const semantic_runtime_descriptor & rt,
        const worker_role role,
        const int32_t layer_end,
        const semantic_blob & blob) {
    if (role != worker_role::pipeline_stage) {
        return false;
    }
    const int32_t n_layer = runtime_layer_count(rt);
    if (n_layer <= 0 || layer_end < n_layer) {
        return false;
    }
    return blob.id == "output_norm" ||
           blob.id == "output_head" ||
           blob.role == tensor_semantic_role::output_norm ||
           blob.role == tensor_semantic_role::output_head;
}

} // namespace

std::vector<std::string> runtime_worker_required_blobs(
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
        } else if (role == worker_role::pipeline_stage && layer_start == 0 &&
                (blob.id == "embedding" || blob.role == tensor_semantic_role::embedding)) {
            // The first pipeline stage graph still builds the token embedding branch during
            // graph reservation, even when runtime requests use external embedding vectors.
            required.push_back(blob.id);
        } else if (final_stage_output_required(rt, role, layer_end, blob)) {
            // The final pipeline stage may execute the local output head; include the output
            // tensors so graph reservation can build result_norm/result_output safely.
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

llama_model * runtime_load_model_from_layer_store(
        layer_store store,
        const model_manifest & manifest,
        const runtime_layer_store_model_load_request & req,
        std::string & err) {
    err.clear();

    const auto metadata = store.load_metadata_blob();
    if (!metadata.has_value() || metadata->empty()) {
        err = "metadata.bin missing in layer store";
        return nullptr;
    }

    const semantic_runtime_descriptor rt = build_semantic_runtime_descriptor(manifest);
    const std::vector<std::string> required_blobs =
            runtime_worker_required_blobs(rt, req.role, req.layer_start, req.layer_end);

    layer_store_tensor_provider provider(std::move(store), rt);
    if (req.verify_tensors && !req.params.vocab_only) {
        for (const std::string & blob_id : required_blobs) {
            const semantic_blob * blob = find_blob(rt.blobs, blob_id);
            if (blob == nullptr) {
                continue;
            }
            for (const semantic_tensor_slot & slot : blob->tensors) {
                if (!provider.tensor_exists(slot.name)) {
                    err = "missing layer store tensor " + slot.name;
                    return nullptr;
                }
            }
        }
    }

    gguf_init_params gparams{};
    gparams.no_alloc = true;
    gguf_context * gguf = gguf_init_from_buffer(metadata->data(), metadata->size(), gparams);
    if (gguf == nullptr) {
        err = "failed to parse metadata.bin as GGUF";
        return nullptr;
    }

    runtime_llama_tensor_load_ctx load_ctx{};
    load_ctx.provider = &provider;
    load_ctx.layer_start = req.layer_start;
    load_ctx.layer_end   = req.layer_end;
    std::unordered_set<std::string> required_tensors;
    for (const std::string & blob_id : required_blobs) {
        const semantic_blob * blob = find_blob(rt.blobs, blob_id);
        if (blob == nullptr) {
            continue;
        }
        for (const semantic_tensor_slot & slot : blob->tensors) {
            required_tensors.insert(slot.name);
        }
    }
    load_ctx.required_tensors = &required_tensors;

    llama_model * model = llama_model_init_from_user_filtered(
            gguf,
            runtime_llama_set_tensor_data,
            &load_ctx,
            runtime_llama_tensor_required,
            &load_ctx,
            req.params);
    gguf_free(gguf);

    if (model == nullptr) {
        err = load_ctx.last_error.empty() ? "llama_model_init_from_user failed" : load_ctx.last_error;
        return nullptr;
    }
    if (!load_ctx.last_error.empty()) {
        llama_model_free(model);
        err = load_ctx.last_error;
        return nullptr;
    }

    return model;
}

void runtime_free_model(llama_model * model) {
    if (model != nullptr) {
        llama_model_free(model);
    }
}
