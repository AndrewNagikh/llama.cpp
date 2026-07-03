#include "runtime_worker_bind.h"

#include "architecture/semantic_runtime_descriptor.h"
#include "layer_store/descriptor_materialize.h"
#include "runtime/layer_store_tensor_provider.h"

#include <filesystem>

namespace {

std::string worker_gguf_path_for_role(const layer_store & store, const worker_role role) {
    const std::string role_name = worker_role_to_string(role);
    return (store.model_root() / ("worker_" + role_name + ".gguf")).string();
}

bool gguf_file_usable(const std::string & path) {
    if (path.empty()) {
        return false;
    }
    std::error_code ec;
    return std::filesystem::exists(path, ec) && std::filesystem::file_size(path, ec) > 0;
}

architecture_descriptor arch_from_rt(const semantic_runtime_descriptor & rt) {
    architecture_descriptor desc;
    desc.architecture     = rt.architecture;
    desc.family           = rt.family;
    desc.tied_embeddings  = rt.tied_embeddings;
    desc.separate_lm_head = rt.separate_lm_head;
    desc.is_moe           = rt.is_moe;
    desc.blobs            = rt.blobs;
    for (const worker_descriptor & w : rt.workers) {
        worker_requirement req;
        req.role           = w.role;
        req.required_blobs = w.required_blob_ids;
        desc.worker_requirements.push_back(std::move(req));
    }
    return desc;
}

} // namespace

bool runtime_verify_worker_tensors(
        const layer_store & store,
        const model_manifest & manifest,
        const worker_role role,
        const int32_t layer_start,
        const int32_t layer_end,
        std::string & err) {
    if (role == worker_role::sampler) {
        err.clear();
        return true;
    }
    if (role == worker_role::tokenizer) {
        if (!store.metadata_bytes().has_value()) {
            err = "metadata.bin missing for tokenizer";
            return false;
        }
        err.clear();
        return true;
    }

    const semantic_runtime_descriptor rt = build_semantic_runtime_descriptor(manifest);
    const worker_materialize_plan plan = worker_materialize_plan_for_role(
            rt, role, layer_start, layer_end);

    if (!plan.required_blobs.empty()) {
        const architecture_descriptor arch = arch_from_rt(rt);
        if (!verify_required_blobs(store, arch, plan.required_blobs, err)) {
            return false;
        }
    }

    const layer_store_tensor_provider provider(store, rt);
    for (const std::string & blob_id : plan.required_blobs) {
        const semantic_blob * blob = find_blob(rt.blobs, blob_id);
        if (blob == nullptr) {
            continue;
        }
        for (const semantic_tensor_slot & slot : blob->tensors) {
            if (!provider.tensor_exists(slot.name)) {
                err = "missing tensor " + slot.name + " for role " + worker_role_to_string(role);
                return false;
            }
        }
    }

    for (int32_t layer = layer_start; layer < layer_end; ++layer) {
        const std::string blob_id = layer_blob_id(layer);
        const semantic_blob * blob = find_blob(rt.blobs, blob_id);
        if (blob == nullptr) {
            err = "missing layer blob " + blob_id;
            return false;
        }
        for (const semantic_tensor_slot & slot : blob->tensors) {
            if (!provider.tensor_exists(slot.name)) {
                err = "missing layer tensor " + slot.name;
                return false;
            }
        }
    }

    err.clear();
    return true;
}

runtime_bind_result runtime_bind_worker(
        const layer_store & store,
        const model_manifest & manifest,
        const worker_role role,
        const int32_t layer_start,
        const int32_t layer_end) {
    runtime_bind_result result{};

    if (role == worker_role::sampler) {
        result.success            = true;
        result.tensors_ready        = true;
        result.materialize_required = false;
        return result;
    }

    std::string verr;
    result.tensors_ready = runtime_verify_worker_tensors(
            store, manifest, role, layer_start, layer_end, verr);
    if (!result.tensors_ready) {
        result.error = verr;
        return result;
    }

    const std::string path = worker_gguf_path_for_role(store, role);
    if (gguf_file_usable(path)) {
        result.success            = true;
        result.cached_gguf_ready    = true;
        result.materialize_required = false;
        result.worker_gguf_path     = path;
        return result;
    }

    result.success            = true;
    result.materialize_required = true;
    return result;
}
