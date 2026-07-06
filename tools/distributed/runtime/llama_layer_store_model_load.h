#pragma once

#include "architecture/semantic_blob.h"
#include "architecture/semantic_runtime_descriptor.h"
#include "architecture/worker_requirement.h"
#include "node_agent/layer_store/layer_store.h"
#include "orchestrator/manifest_builder/manifest_builder.h"

#include "llama.h"

#include <algorithm>
#include <string>
#include <vector>

// Task 11.7 — load llama model weights from Layer Store via TensorProvider + metadata.bin.

struct runtime_layer_store_model_load_request {
    worker_role role           = worker_role::pipeline_stage;
    int32_t     layer_start    = 0;
    int32_t     layer_end      = 0;
    llama_model_params params  = llama_model_default_params();
    bool        verify_tensors = true;
};

std::vector<std::string> runtime_worker_required_blobs(
        const semantic_runtime_descriptor & rt,
        const worker_role role,
        const int32_t layer_start,
        const int32_t layer_end);

llama_model * runtime_load_model_from_layer_store(
        layer_store store,
        const model_manifest & manifest,
        const runtime_layer_store_model_load_request & req,
        std::string & err);

void runtime_free_model(llama_model * model);
