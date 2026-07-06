#pragma once

#include "architecture/worker_requirement.h"
#include "dist_common.h"
#include "node_agent/layer_store/layer_store.h"
#include "runtime/runtime_worker_bind.h"

#include <functional>
#include <string>

// Task 11.7/11.8 — bind Layer Store tensors, optionally assemble compat worker GGUF.

struct runtime_worker_gguf_request {
    dist_configure_req cfg;
    nlohmann::json     body;
    bool               verify_materialization = false;
    std::function<bool(const std::string & path, worker_role role)> cache_shell_ok;
};

struct runtime_worker_gguf_resolve_result {
    std::string          worker_gguf_path;
    runtime_bind_result  bind;
    worker_role          materialize_role = worker_role::pipeline_stage;
    int32_t              layer_start      = 0;
    int32_t              layer_end        = 0;
    bool                 from_cache       = false;
    bool                 compat_materialized = false;
};

runtime_worker_gguf_resolve_result runtime_resolve_worker_gguf(
        layer_store store,
        const runtime_worker_gguf_request & req,
        std::string & err);
