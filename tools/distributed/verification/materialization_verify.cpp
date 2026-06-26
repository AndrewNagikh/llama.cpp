#include "materialization_verify.h"

#include "node_agent/layer_store/layer_gguf_assembler.h"
#include "node_agent/layer_store/layer_special.h"
#include "verification_common.h"

#include <filesystem>

bool materialize_full_model(
        const layer_store & store,
        const model_manifest & manifest,
        const std::string & output_path) {
    if (manifest.n_layer <= 0) {
        return false;
    }
    return layer_store_materialize_gguf(
            store,
            manifest,
            output_path,
            0,
            static_cast<int32_t>(manifest.n_layer),
            true,
            true);
}

materialization_verify_result verify_materialization_repeatability(
        const layer_store & store,
        const model_manifest & manifest,
        const std::string & output_dir) {
    materialization_verify_result result;
    result.summary.name = "materialization_repeatability";

    std::error_code ec;
    std::filesystem::create_directories(output_dir, ec);

    const std::string path_a = output_dir + "/materialized_a.gguf";
    const std::string path_b = output_dir + "/materialized_b.gguf";

    if (!materialize_full_model(store, manifest, path_a) ||
            !materialize_full_model(store, manifest, path_b)) {
        result.summary.status  = verify_status::fail;
        result.summary.message = "materialization failed";
        return result;
    }

    result.sha256_repeat      = sha256_file_hex(path_a);
    result.sha256_materialized = result.sha256_repeat;
    const std::string sha_b   = sha256_file_hex(path_b);

    if (result.sha256_repeat.empty() || sha_b.empty()) {
        result.summary.status  = verify_status::fail;
        result.summary.message = "failed to hash materialized outputs";
        return result;
    }

    const bool ok = (result.sha256_repeat == sha_b);
    result.summary.status  = ok ? verify_status::ok : verify_status::fail;
    result.summary.message = ok ? "repeatable materialization (SHA256 match)" : "non-deterministic materialization";
    result.summary.details = {
        { "sha256_a", result.sha256_repeat },
        { "sha256_b", sha_b },
    };
    return result;
}
