#include "verification/layer_verify.h"
#include "verification/layer_store_populate.h"

#include "orchestrator/manifest_builder/manifest_builder.h"
#include "test_manifest_common.h"

#include <cstdio>
#include <filesystem>

int main(int argc, char ** argv) {
    std::string source = manifest_test_model_path();
    std::string store_root = "/tmp/layer_verify_store";
    if (argc >= 2) {
        source = argv[1];
    }
    if (argc >= 3) {
        store_root = argv[2];
    }
    if (source.empty() || !std::filesystem::exists(source)) {
        fprintf(stderr, "layer_verify: model not found\n");
        return 77;
    }

    const auto built = build_manifest_from_file(source);
    if (!built.success) {
        fprintf(stderr, "layer_verify: manifest failed\n");
        return 1;
    }

    layer_store store(store_root, "verify-model");
    if (!populate_layer_store_from_gguf(store, built.manifest, source)) {
        fprintf(stderr, "layer_verify: populate failed\n");
        return 1;
    }

    const auto result = layer_verify_store(store, built.manifest, source);
    nlohmann::json out = result.summary.to_json();
    out["issues"] = result.issues;
    printf("%s\n", out.dump(2).c_str());
    return result.summary.status == verify_status::ok ? 0 : 1;
}
