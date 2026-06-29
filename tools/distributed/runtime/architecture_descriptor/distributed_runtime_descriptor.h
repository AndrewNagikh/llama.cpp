#pragma once

#include "architecture/semantic_runtime_descriptor.h"
#include "blob_requirements.h"

#include "nlohmann/json.hpp"

#include <map>
#include <string>

// Task 9.9 — runtime-facing architecture descriptor (family-agnostic contract).

struct runtime_capabilities {
    bool supports_partial_forward     = true;
    bool supports_hidden_injection    = true;
    bool supports_embedding_injection = true;
};

struct distributed_runtime_descriptor {
    semantic_runtime_descriptor semantic;

    runtime_capabilities capabilities;

    // blob_id → stage requirements (ENTRY / MIDDLE / FINAL).
    std::map<std::string, blob_stage_requirements> blob_stages;

    bool empty() const { return semantic.empty(); }

    nlohmann::json to_json() const;
};

distributed_runtime_descriptor build_distributed_runtime_descriptor(
        const model_manifest & manifest);
