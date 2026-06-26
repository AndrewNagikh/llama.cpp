#pragma once

#include "layer_store/layer_store.h"
#include "manifest_builder/manifest_builder.h"
#include "verification_types.h"

#include <string>

verification_report run_verification_pipeline(
        const std::string & model_id,
        const std::string & original_gguf,
        const layer_store & store,
        const model_manifest & manifest,
        const std::string & work_dir);
