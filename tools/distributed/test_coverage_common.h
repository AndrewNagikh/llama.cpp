#pragma once

#include "orchestrator/coverage/coverage.h"
#include "orchestrator/layout_planner/layout_planner.h"
#include "test_layout_common.h"

#include <string>
#include <vector>

inline desired_model_layout make_desired_layout(
        const std::string & model_id,
        const std::vector<std::pair<std::string, int32_t>> & assignments) {
    desired_model_layout desired;
    desired.model_id = model_id;
    desired.fits_cluster = true;
    for (const auto & item : assignments) {
        layer_placement p;
        p.layer_index = item.second;
        p.node_id     = item.first;
        p.device      = "cuda";
        p.size_bytes  = 64 * 1024 * 1024;
        p.required    = true;
        desired.placements.push_back(p);
    }
    desired.total_weight_bytes = static_cast<uint64_t>(desired.placements.size()) * 64 * 1024 * 1024;
    return desired;
}

inline installed_layer make_ready_layer(
        int32_t layer_index,
        const std::string & node_id,
        const std::string & checksum = "") {
    installed_layer layer;
    layer.layer_index = layer_index;
    layer.node_id     = node_id;
    layer.device      = "cuda";
    layer.size_bytes  = 64 * 1024 * 1024;
    layer.checksum    = checksum.empty()
            ? ("stub:layer:" + std::to_string(layer_index))
            : checksum;
    layer.state       = install_state::ready;
    layer.updated_at  = std::chrono::system_clock::now();
    return layer;
}

inline actual_model_layout make_actual_layout(
        const std::string & model_id,
        const std::vector<installed_layer> & layers) {
    actual_model_layout actual;
    actual.model_id = model_id;
    actual.layers   = layers;
    return actual;
}
