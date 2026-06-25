#include "orchestrator/manifest_builder/manifest_builder.h"
#include "orchestrator/model_registry.h"
#include "test_manifest_common.h"

#include <cstdio>
#include <filesystem>
#include <string>

int main() {
    const std::string path = manifest_test_model_path();
    if (path.empty() || !std::filesystem::exists(path)) {
        fprintf(stderr, "test-manifest-refresh: GGUF model not found (set MODEL env)\n");
        return 77;
    }

    cluster_model_registry reg;

    dist_model_record record;
    record.model_id  = "refresh-test";
    record.filename  = std::filesystem::path(path).filename().string();
    record.status    = dist_model_status::manifest_pending;
    reg.add_or_update(record);

    const std::string models_dir = std::filesystem::path(path).parent_path().string();

    const auto first = build_manifest_for_record(record, models_dir, path);
    if (!first.success || first.manifest.n_layer == 0 || first.manifest.tensors.empty()) {
        fprintf(stderr, "test-manifest-refresh: first build failed: %s\n", first.error.c_str());
        return 1;
    }
    if (!reg.apply_manifest("refresh-test", first.manifest)) {
        fprintf(stderr, "test-manifest-refresh: apply_manifest failed\n");
        return 1;
    }

    const auto * stored = reg.find("refresh-test");
    if (!stored || stored->status != dist_model_status::manifest_ready || !stored->manifest) {
        fprintf(stderr, "test-manifest-refresh: registry state invalid after first apply\n");
        return 1;
    }
    if (stored->manifest->n_layer != first.manifest.n_layer) {
        fprintf(stderr, "test-manifest-refresh: n_layer mismatch after first apply\n");
        return 1;
    }

    const auto second = build_manifest_for_record(record, models_dir, path);
    if (!second.success) {
        fprintf(stderr, "test-manifest-refresh: second build failed: %s\n", second.error.c_str());
        return 1;
    }
    if (!reg.apply_manifest("refresh-test", second.manifest)) {
        fprintf(stderr, "test-manifest-refresh: second apply_manifest failed\n");
        return 1;
    }

    const auto * refreshed = reg.find("refresh-test");
    if (!refreshed || !refreshed->manifest) {
        fprintf(stderr, "test-manifest-refresh: registry state invalid after refresh\n");
        return 1;
    }
    if (refreshed->manifest->tensors.size() != first.manifest.tensors.size() ||
        refreshed->manifest->layers.size() != first.manifest.layers.size()) {
        fprintf(stderr, "test-manifest-refresh: manifest size changed unexpectedly\n");
        return 1;
    }

    printf("test-manifest-refresh: OK n_layer=%u tensors=%zu\n",
            refreshed->manifest->n_layer,
            refreshed->manifest->tensors.size());
    return 0;
}
