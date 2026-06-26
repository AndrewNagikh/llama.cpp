#pragma once

#include "orchestrator/install_planner/install_planner.h"
#include "test_coverage_common.h"

#include <cstdint>
#include <string>

inline model_manifest make_manifest_with_layer_ranges(
        int32_t n_layer,
        uint64_t bytes_per_layer,
        uint64_t base_offset = 4096) {
    model_manifest manifest;
    manifest.n_layer = static_cast<uint32_t>(n_layer);
    manifest.architecture = "llama";

    for (int32_t i = 0; i < n_layer; ++i) {
        layer_descriptor ld;
        ld.layer_index = i;
        ld.size_bytes  = bytes_per_layer;
        ld.tensors.push_back("blk." + std::to_string(i) + ".attn.weight");
        manifest.layers.push_back(ld);

        tensor_descriptor td;
        td.name       = "blk." + std::to_string(i) + ".attn.weight";
        td.layer      = i;
        td.offset     = base_offset + static_cast<uint64_t>(i) * bytes_per_layer;
        td.size_bytes = bytes_per_layer;
        td.role       = tensor_role::layer;
        manifest.tensors.push_back(td);
    }
    return manifest;
}

inline coverage_report make_coverage(
        const std::string & model_id,
        const std::vector<int32_t> & missing,
        const std::vector<int32_t> & corrupted,
        int total_layers) {
    coverage_report report;
    report.model_id = model_id;
    report.total_layers = total_layers;
    report.missing = missing;
    report.corrupted = corrupted;
    report.missing_layers = static_cast<int>(missing.size());
    report.corrupted_layers = static_cast<int>(corrupted.size());
    report.ready_layers = total_layers - report.missing_layers - report.corrupted_layers;
    if (report.corrupted_layers > 0) {
        report.state = coverage_state::degraded;
    } else if (report.missing_layers == 0) {
        report.state = coverage_state::ready;
    } else if (report.ready_layers == 0) {
        report.state = coverage_state::empty;
    } else {
        report.state = coverage_state::partial;
    }
    return report;
}
