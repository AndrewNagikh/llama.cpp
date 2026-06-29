#pragma once

#include "coverage/coverage.h"
#include "layer_store/layer_store.h"
#include "manifest_builder/manifest_builder.h"

#include "architecture/semantic_runtime_descriptor.h"

#include "nlohmann/json.hpp"

#include <string>
#include <vector>

// Task 9.9 — verify local Layer Store against manifest + registry view.

struct store_blob_entry {
    std::string blob_id;
    std::string tensor_name;
    std::string checksum;
    uint64_t    size_bytes = 0;
    std::string node_id;
    install_state state = install_state::missing;
};

struct store_verify_result {
    bool ok = false;
    std::string node_id;
    int  blob_count = 0;
    int  verified_count = 0;
    int  missing_count = 0;
    int  corrupted_count = 0;
    std::vector<std::string> issues;
    std::vector<store_blob_entry> blobs;

    nlohmann::json to_json() const;
};

store_verify_result verify_layer_store(
        const layer_store & store,
        const model_manifest & manifest,
        const std::string & node_id = "");

nlohmann::json summarize_layer_store(const layer_store & store, const std::string & node_id);

store_verify_result compare_store_to_actual(
        const store_verify_result & store_view,
        const actual_model_layout & actual,
        const std::string & node_id);
