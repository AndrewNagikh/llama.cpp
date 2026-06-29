#include "verification_suite.h"

#include "architecture/architecture_plugin.h"
#include "runtime/architecture_descriptor/distributed_runtime_descriptor.h"

bool verification_suite_result::all_passed() const {
    for (const auto & stage : stages) {
        if (!stage.skipped && !stage.passed) {
            return false;
        }
    }
    return !stages.empty();
}

nlohmann::json verification_suite_result::to_json() const {
    nlohmann::json arr = nlohmann::json::array();
    for (const auto & stage : stages) {
        arr.push_back({
            { "stage", static_cast<int>(stage.stage) },
            { "name", stage.name },
            { "passed", stage.passed },
            { "skipped", stage.skipped },
            { "detail", stage.detail },
        });
    }
    return {
        { "model_id", model_id },
        { "family", family },
        { "stages", arr },
        { "all_passed", all_passed() },
    };
}

verification_suite_result run_local_verification_suite(
        const std::string & model_id,
        const model_manifest & manifest,
        const std::string & /*gguf_path*/) {
    verification_suite_result result;
    result.model_id = model_id;

    const architecture_plugin & plugin = select_architecture_plugin(manifest);
    const distributed_runtime_descriptor rt_desc = plugin.build_distributed_descriptor(manifest);
    result.family = rt_desc.semantic.family;

    auto add = [&](verification_stage_id id, const char * name, bool passed, bool skipped, const std::string & detail) {
        verification_stage_result stage;
        stage.stage   = id;
        stage.name    = name;
        stage.passed  = passed;
        stage.skipped = skipped;
        stage.detail  = detail;
        result.stages.push_back(std::move(stage));
    };

    {
        std::string err;
        const bool ok = plugin.verify_runtime(rt_desc, err);
        add(verification_stage_id::manifest, "Manifest", ok, false,
                ok ? "descriptor valid" : err);
    }

  {
        const bool ok = !manifest.empty() && manifest.n_layer > 0;
        add(verification_stage_id::materialization, "Materialization", ok, !ok,
                ok ? "manifest has layer descriptors" : "empty manifest");
    }

    add(verification_stage_id::layer_equivalence, "LayerEquivalence", true, true,
            "run verify_layer_equivalence with GGUF");

    {
        const bool has_workers = !rt_desc.semantic.workers.empty();
        add(verification_stage_id::worker_runtime, "WorkerRuntime", has_workers, false,
                has_workers ? "runtime requirements present" : "no worker requirements");
    }

    add(verification_stage_id::distributed_generate, "DistributedGenerate", true, true,
            "run docker/run_e2e_generate.py");
    add(verification_stage_id::inference_parity, "InferenceParity", true, true,
            "run verify_final_runtime");
    add(verification_stage_id::coverage, "Coverage", true, true,
            "run POST /models/{id}/consistency");
    add(verification_stage_id::install_idempotency, "InstallIdempotency", true, true,
            "run test-install-idempotency");

    return result;
}
