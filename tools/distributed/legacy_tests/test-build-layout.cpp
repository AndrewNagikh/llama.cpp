#include "orchestrator/layout_planner/layout_planner.h"
#include "orchestrator/manifest_builder/manifest_builder.h"
#include "orchestrator/model_registry.h"
#include "test_layout_common.h"

#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <string>

static bool run_pipeline(cluster_model_registry & reg, const std::string & model_id,
        const model_manifest & manifest, const std::vector<layout_node_input> & nodes,
        std::string & err) {
    if (!reg.apply_manifest(model_id, manifest, nullptr)) {
        err = "apply_manifest failed";
        return false;
    }

    const auto built = build_desired_layout(model_id, manifest, nodes);
    if (!built.success) {
        err = "build_desired_layout failed: " + built.error;
        return false;
    }
    if (!built.layout.fits_cluster) {
        err = "fits_cluster is false";
        return false;
    }

    if (!reg.apply_layout(model_id, built.layout, nullptr)) {
        err = "apply_layout failed";
        return false;
    }

    const auto * record = reg.find(model_id);
    if (!record || !record->layout.has_value()) {
        err = "layout not stored in registry";
        return false;
    }

    std::string verr;
    if (!validate_desired_layout(record->layout->desired, manifest, verr)) {
        err = "validation failed: " + verr;
        return false;
    }
    return true;
}

int main() {
    const char * model_env = std::getenv("MODEL");
    if (!model_env || !std::filesystem::exists(model_env)) {
        std::cerr << "test-build-layout: set MODEL to a local GGUF file\n";
        return 77;
    }

    const model_manifest manifest = build_manifest_from_file(model_env);
    if (manifest.empty() || manifest.layers.empty()) {
        std::cerr << "test-build-layout: manifest parse failed\n";
        return 1;
    }

    const std::vector<layout_node_input> nodes = {
        make_layout_node("node-a", 500.0, 32.0, 24.0, "cuda"),
        make_layout_node("node-b", 300.0, 32.0, 24.0, "metal"),
        make_layout_node("node-c", 100.0, 32.0, 0.0),
    };

    cluster_model_registry reg;
    dist_model_record record;
    record.model_id = "llama-3.2-1b";
    record.status   = dist_model_status::manifest_pending;
    reg.add_or_update(record);

    std::string err;
    if (!run_pipeline(reg, "llama-3.2-1b", manifest, nodes, err)) {
        std::cerr << "test-build-layout: " << err << "\n";
        return 1;
    }

    const auto * stored = reg.find("llama-3.2-1b");
    std::cout << "test-build-layout: OK placements="
              << stored->layout->desired.placements.size()
              << " n_layer=" << manifest.n_layer << "\n";
    return 0;
}
