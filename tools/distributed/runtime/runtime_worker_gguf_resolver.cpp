#include "runtime_worker_gguf_resolver.h"

#include "architecture/semantic_runtime_descriptor.h"
#include "node_agent/layer_store/layer_gguf_assembler.h"
#include "runtime/runtime_config.h"
#include "runtime/runtime_compat_materialize.h"
#include "runtime/runtime_role.h"
#include "verification/worker_verify.h"

#include <filesystem>

namespace {

worker_role dist_role_to_worker_role(const dist_node_role role) {
    switch (role) {
        case DIST_ROLE_ENTRY:
        case DIST_ROLE_MIDDLE:
        case DIST_ROLE_FINAL:
            return worker_role::pipeline_stage;
        default:
            return worker_role::pipeline_stage;
    }
}

worker_role resolve_materialize_role(
        const dist_configure_req & cfg,
        const nlohmann::json & body,
        int32_t & mat_layer_start,
        int32_t & mat_layer_end) {
    (void) mat_layer_start;
    (void) mat_layer_end;

    worker_role role = dist_role_to_worker_role(cfg.role);
    if (body.contains("runtime_role")) {
        const runtime_role rt = runtime_role_from_string(body.value("runtime_role", ""));
        if (rt != runtime_role::unassigned) {
            role = worker_role_from_runtime_role(rt);
        }
    }
    return role;
}

bool compat_materialize_role_shell(
        const layer_store & store,
        const model_manifest & manifest,
        const semantic_runtime_descriptor & rt,
        const worker_role role,
        const int32_t mat_layer_start,
        const int32_t mat_layer_end,
        const std::string & out_path,
        std::string & err) {
    if (role == worker_role::tokenizer) {
        return runtime_compat_materialize_worker_gguf(
                store, manifest, rt, worker_role::tokenizer, 0, 0, out_path, err);
    }
    if (role == worker_role::embedding) {
        return runtime_compat_materialize_worker_gguf(
                store, manifest, rt, worker_role::embedding, 0, 0, out_path, err);
    }
    if (role == worker_role::output_head) {
        return runtime_compat_materialize_worker_gguf(
                store,
                manifest,
                rt,
                worker_role::output_head,
                0,
                0,
                out_path,
                err);
    }
    return runtime_compat_materialize_worker_gguf(
            store, manifest, rt, role, mat_layer_start, mat_layer_end, out_path, err);
}

} // namespace

runtime_worker_gguf_resolve_result runtime_resolve_worker_gguf(
        layer_store store,
        const runtime_worker_gguf_request & req,
        std::string & err) {
    runtime_worker_gguf_resolve_result result{};

    if (req.cfg.model_id.empty()) {
        err = "model_id required for layer-first configure";
        return result;
    }

    const auto manifest = store.load_manifest();
    if (!manifest.has_value()) {
        err = "manifest not found in layer store; run install/sync first";
        return result;
    }

    if (!store.metadata_bytes().has_value() && !req.cfg.source_url.empty()) {
        layer_store_cache_metadata(store, *manifest, req.cfg.source_url);
    }

    result.layer_start = req.cfg.layer_start;
    result.layer_end   = req.cfg.layer_end;
    result.materialize_role =
            resolve_materialize_role(req.cfg, req.body, result.layer_start, result.layer_end);

    if (result.materialize_role == worker_role::sampler) {
        result.bind.success            = true;
        result.bind.tensors_ready        = true;
        result.bind.materialize_required = false;
        return result;
    }

    if (result.materialize_role != worker_role::tokenizer && !store.metadata_bytes().has_value()) {
        err = "metadata.bin missing in layer store; run install/sync first";
        return result;
    }

    const semantic_runtime_descriptor rt = build_semantic_runtime_descriptor(*manifest);

    if (req.verify_materialization) {
        std::string verr;
        if (!verify_worker_materialization_for_role(
                    store,
                    *manifest,
                    result.materialize_role,
                    result.layer_start,
                    result.layer_end,
                    verr)) {
            err = "materialization verification failed: " + verr;
            return result;
        }
    }

    const bool force_materialize = req.body.value("force_materialize", false) ||
            runtime_force_materialize();
    const bool bind_only = req.body.value("bind_only", false) || runtime_bind_only_request();

    result.bind = runtime_bind_worker(
            store, *manifest, result.materialize_role, result.layer_start, result.layer_end);
    if (!result.bind.success && !result.bind.tensors_ready) {
        err = result.bind.error.empty() ? "layer store bind failed" : result.bind.error;
        return result;
    }

    const std::string role_name = worker_role_to_string(result.materialize_role);

    if (!force_materialize && runtime_layer_first_enabled() && result.bind.tensors_ready) {
        fprintf(stderr,
                "node_agent: layer-store bind role=%s layers=[%d,%d) (no gguf on inference path)\n",
                role_name.c_str(),
                result.layer_start,
                result.layer_end);
        return result;
    }

    if (!force_materialize && result.bind.cached_gguf_ready && !result.bind.worker_gguf_path.empty()) {
        const bool cache_ok = !req.cache_shell_ok ||
                req.cache_shell_ok(result.bind.worker_gguf_path, result.materialize_role);
        if (cache_ok) {
            fprintf(stderr,
                    "node_agent: using cached %s role=%s layers=[%d,%d) bind=layer_store\n",
                    result.bind.worker_gguf_path.c_str(),
                    role_name.c_str(),
                    result.layer_start,
                    result.layer_end);
            result.worker_gguf_path = result.bind.worker_gguf_path;
            result.from_cache         = true;
            return result;
        }
        std::error_code ec;
        std::filesystem::remove(result.bind.worker_gguf_path, ec);
        fprintf(stderr,
                "node_agent: stale cached %s for role=%s, rematerializing\n",
                result.bind.worker_gguf_path.c_str(),
                role_name.c_str());
    }

    if (bind_only) {
        if (!result.bind.tensors_ready) {
            err = result.bind.error.empty() ? "tensors not ready in layer store" : result.bind.error;
            return result;
        }
        if (!result.bind.materialize_required && !result.bind.worker_gguf_path.empty()) {
            result.worker_gguf_path = result.bind.worker_gguf_path;
            result.from_cache         = true;
            return result;
        }
        if (runtime_layer_first_enabled()) {
            err.clear();
            return result;
        }
        err = "bind_only: compat GGUF not cached and materialize skipped";
        return result;
    }

    if (runtime_layer_first_enabled() && result.bind.tensors_ready &&
            !result.bind.materialize_required && !result.bind.worker_gguf_path.empty()) {
        result.worker_gguf_path = result.bind.worker_gguf_path;
        result.from_cache         = true;
        return result;
    }

    const std::string out_path =
            (store.model_root() / ("worker_" + role_name + ".gguf")).string();

    if (!compat_materialize_role_shell(
                store,
                *manifest,
                rt,
                result.materialize_role,
                result.layer_start,
                result.layer_end,
                out_path,
                err)) {
        if (err.empty()) {
            err = "failed to assemble compat GGUF for role " + role_name;
        }
        return result;
    }

    result.worker_gguf_path      = out_path;
    result.compat_materialized = true;
    result.bind.worker_gguf_path = out_path;
    result.bind.cached_gguf_ready    = true;
    result.bind.materialize_required = false;

    fprintf(stderr,
            "node_agent: compat materialized %s role=%s layers=[%d,%d)\n",
            out_path.c_str(),
            role_name.c_str(),
            result.layer_start,
            result.layer_end);
    return result;
}
