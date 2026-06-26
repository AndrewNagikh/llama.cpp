#include "verification_stages.h"

#include "alignment_verify.h"
#include "gguf_diff.h"
#include "hidden_state_verify.h"
#include "layer_verify.h"
#include "load_verify.h"
#include "logits_compare.h"
#include "manifest_verify.h"
#include "materialization_verify.h"
#include "metadata_verify.h"
#include "parity_verify.h"
#include "tensor_verify.h"
#include "verification_common.h"

#include <filesystem>

static void push_stage(
        stage_pipeline_report & report,
        const verify_check_result & stage,
        const stage_pipeline_options & options) {
    report.stages.push_back(stage);
    if (stage.status == verify_status::fail && report.failed_stage.empty()) {
        report.failed_stage = stage.name;
    }
    if (options.stop_on_first_fail && stage.status == verify_status::fail) {
        report.passed = false;
    }
}

static bool should_continue(
        const stage_pipeline_report & report,
        const stage_pipeline_options & options) {
    return !options.stop_on_first_fail || report.failed_stage.empty();
}

nlohmann::json stage_pipeline_report::to_json() const {
    nlohmann::json stages_json = nlohmann::json::array();
    for (const auto & s : stages) {
        stages_json.push_back(s.to_json());
    }
    return {
        { "model_id", model_id },
        { "original_path", original_path },
        { "materialized_path", materialized_path },
        { "passed", passed },
        { "failed_stage", failed_stage },
        { "stages", stages_json },
    };
}

stage_pipeline_report run_materialization_verification_stages(
        const std::string & model_id,
        const std::string & original_gguf,
        const layer_store & store,
        const model_manifest & manifest,
        const std::string & work_dir,
        const stage_pipeline_options & options) {
    stage_pipeline_report report;
    report.model_id      = model_id;
    report.original_path = original_gguf;

    std::error_code ec;
    std::filesystem::create_directories(work_dir, ec);

    const worker_tensor_plan full_plan = make_worker_tensor_plan(
            worker_verify_role::full,
            0,
            static_cast<int32_t>(manifest.n_layer));

    // Stage 1 — Manifest vs original GGUF
    {
        const auto m = manifest_verify_against_gguf(manifest, original_gguf);
        push_stage(report, m.summary, options);
    }
    if (!should_continue(report, options)) {
        return report;
    }

    // Stage 2 — Layer store blobs
    {
        const auto layers = layer_verify_store(store, manifest, original_gguf);
        push_stage(report, layers.summary, options);
    }
    if (!should_continue(report, options)) {
        return report;
    }

    // Stage 3 — Materializer input (FULL worker plan)
    {
        const auto plan_check = verify_worker_tensor_plan(manifest, full_plan);
        push_stage(report, plan_check, options);
    }
    if (!should_continue(report, options)) {
        return report;
    }

    // Materialize before stages 4+
    const std::string mat_path = work_dir + "/materialized.gguf";
    if (!materialize_full_model(store, manifest, mat_path)) {
        verify_check_result fail;
        fail.name    = "materialize";
        fail.status  = verify_status::fail;
        fail.message = "full materialize failed";
        push_stage(report, fail, options);
        return report;
    }
    report.materialized_path = mat_path;

    // Stage 4 — Materialized GGUF structure
    {
        const auto structure = verify_materialized_structure_for_role(mat_path, full_plan, manifest);
        push_stage(report, structure, options);
    }
    if (!should_continue(report, options)) {
        return report;
    }

    // Stage 5 — GGUF diff (header, metadata, tensor directory)
    {
        const auto diff = gguf_diff_files(original_gguf, mat_path);
        push_stage(report, diff.header, options);
        if (should_continue(report, options)) {
            push_stage(report, diff.metadata, options);
        }
        if (should_continue(report, options)) {
            push_stage(report, diff.tensor_directory, options);
        }
    }
    if (!should_continue(report, options)) {
        return report;
    }

    // Stage 6 — Per-tensor SHA256
    {
        const auto tensors = tensor_verify_files(original_gguf, mat_path);
        push_stage(report, tensors.summary, options);
    }
    if (!should_continue(report, options)) {
        return report;
    }

    // Stage 7 — Worker assignment (full store)
    {
        const auto assign = verify_worker_store_assignment(store, manifest, full_plan);
        push_stage(report, assign, options);
    }
    if (!should_continue(report, options)) {
        return report;
    }

    // Alignment + repeatability (supporting checks)
    {
        const auto align = alignment_verify_file(mat_path);
        push_stage(report, align.summary, options);
    }
    if (should_continue(report, options)) {
        const auto repeat = verify_materialization_repeatability(store, manifest, work_dir + "/repeat");
        push_stage(report, repeat.summary, options);
    }
    if (!should_continue(report, options)) {
        return report;
    }

    // Stage 8 — Runtime load
    {
        const auto load = load_verify_gguf(mat_path);
        push_stage(report, load.summary, options);
    }
    if (!should_continue(report, options)) {
        return report;
    }

    // Stage 9 — Hidden state (primary pass criterion)
    {
        const auto hidden = compare_hidden_states(original_gguf, mat_path, options.prompt);
        push_stage(report, hidden.summary, options);
    }
    if (!should_continue(report, options)) {
        return report;
    }

    // Stage 10 — Logits (primary pass criterion)
    {
        const auto logits = compare_logits_files(original_gguf, mat_path, options.prompt);
        push_stage(report, logits.summary, options);
    }
    if (!should_continue(report, options)) {
        return report;
    }

    // Stage 11 — Sampling (diagnostic; does not block primary pass)
    if (options.run_sampling) {
        const auto parity = verify_inference_parity(
                original_gguf,
                mat_path,
                options.prompt,
                options.max_tokens);
        auto sampling = parity.sampling;
        sampling.name = "stage_11_sampling";
        report.stages.push_back(sampling);
    }

    bool primary_ok = true;
    for (const auto & s : report.stages) {
        if (s.name == "stage_11_sampling") {
            continue;
        }
        if (s.status == verify_status::fail) {
            primary_ok = false;
            if (report.failed_stage.empty()) {
                report.failed_stage = s.name;
            }
            break;
        }
    }

    report.passed = primary_ok;
    return report;
}
