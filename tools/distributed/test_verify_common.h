#pragma once

#include "orchestrator/manifest_builder/manifest_builder.h"
#include "test_manifest_common.h"
#include "verification/layer_store_populate.h"
#include "verification/materialization_verify.h"

#include "node_agent/layer_store/layer_store.h"

#include <filesystem>
#include <string>

#if !defined(_WIN32)
#include <unistd.h>
#endif

inline std::string verify_test_work_dir(const char * name) {
    const auto base = std::filesystem::temp_directory_path() /
            ("dist-verify-" + std::string(name) + "-" + std::to_string(getpid()));
    std::error_code ec;
    std::filesystem::create_directories(base, ec);
    return base.string();
}

inline bool verify_setup_store_from_model(
        const std::string & model_path,
        layer_store & store,
        model_manifest & manifest) {
    manifest = build_manifest_from_file(model_path);
    if (manifest.empty()) {
        return false;
    }
    return populate_layer_store_from_gguf(store, manifest, model_path);
}

inline std::string verify_materialize_from_store(
        const layer_store & store,
        const model_manifest & manifest,
        const std::string & work_dir) {
    const std::string out = work_dir + "/materialized.gguf";
    if (!materialize_full_model(store, manifest, out)) {
        return {};
    }
    return out;
}
