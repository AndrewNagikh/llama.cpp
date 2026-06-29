#pragma once

#include "manifest_builder/manifest_builder.h"

#include "nlohmann/json.hpp"

#include <string>
#include <vector>

// Task 9.9 — unified verification pipeline (8 stages).

enum class verification_stage_id {
    manifest            = 1,
    materialization     = 2,
    layer_equivalence   = 3,
    worker_runtime      = 4,
    distributed_generate = 5,
    inference_parity    = 6,
    coverage            = 7,
    install_idempotency = 8,
};

struct verification_stage_result {
    verification_stage_id stage = verification_stage_id::manifest;
    std::string name;
    bool passed = false;
    bool skipped = false;
    std::string detail;
};

struct verification_suite_result {
    std::string model_id;
    std::string family;
    std::vector<verification_stage_result> stages;
    bool all_passed() const;
    nlohmann::json to_json() const;
};

verification_suite_result run_local_verification_suite(
        const std::string & model_id,
        const model_manifest & manifest,
        const std::string & gguf_path = "");
