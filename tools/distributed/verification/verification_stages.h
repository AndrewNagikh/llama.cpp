#pragma once

#include "layer_store/layer_store.h"
#include "manifest_builder/manifest_builder.h"
#include "verification_types.h"
#include "worker_tensor_plan.h"

#include <string>
#include <vector>

struct stage_pipeline_options {
    std::string prompt           = "The capital of France is";
    int         max_tokens       = 32;
    bool        stop_on_first_fail = true;
    bool        run_sampling     = true;
};

struct stage_pipeline_report {
    std::string model_id;
    std::string original_path;
    std::string materialized_path;
    bool        passed             = false;
    std::string failed_stage;
    std::vector<verify_check_result> stages;

    nlohmann::json to_json() const;
};

stage_pipeline_report run_materialization_verification_stages(
        const std::string & model_id,
        const std::string & original_gguf,
        const layer_store & store,
        const model_manifest & manifest,
        const std::string & work_dir,
        const stage_pipeline_options & options = {});
