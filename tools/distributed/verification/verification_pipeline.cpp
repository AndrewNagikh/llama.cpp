#include "verification_pipeline.h"

#include "alignment_verify.h"
#include "gguf_diff.h"
#include "layer_verify.h"
#include "materialization_verify.h"
#include "metadata_verify.h"
#include "parity_verify.h"
#include "tensor_verify.h"
#include "verification_common.h"

#include <filesystem>

verification_report run_verification_pipeline(
        const std::string & model_id,
        const std::string & original_gguf,
        const layer_store & store,
        const model_manifest & manifest,
        const std::string & work_dir) {
    verification_report report;
    report.model_id      = model_id;
    report.original_path = original_gguf;

    std::error_code ec;
    std::filesystem::create_directories(work_dir, ec);

    const std::string mat_path = work_dir + "/materialized.gguf";
    if (!materialize_full_model(store, manifest, mat_path)) {
        report.passed = false;
        report.materialization_repeatability.status  = verify_status::fail;
        report.materialization_repeatability.message = "full materialize failed";
        return report;
    }
    report.materialized_path = mat_path;

    const auto diff = gguf_diff_files(original_gguf, mat_path);
    report.header            = diff.header;
    report.metadata            = diff.metadata;
    report.tensor_directory    = diff.tensor_directory;
    report.diffs             = diff.differences;

    const auto meta_kv = metadata_verify_kv_prefixes(original_gguf, mat_path);
    if (meta_kv.summary.status == verify_status::fail) {
        report.metadata.status  = verify_status::fail;
        report.metadata.message = meta_kv.summary.message;
        for (const auto & d : meta_kv.differences) {
            report.diffs.push_back("metadata: " + d);
        }
    }

    const auto tensors = tensor_verify_files(original_gguf, mat_path);
    report.tensor_checksums = tensors.summary;
    for (const auto & m : tensors.mismatches) {
        report.diffs.push_back("tensor: " + m);
    }

    const auto align = alignment_verify_file(mat_path);
    report.alignment = align.summary;
    for (const auto & i : align.issues) {
        report.diffs.push_back("alignment: " + i);
    }

    const auto layers = layer_verify_store(store, manifest, original_gguf);
    report.layer_store = layers.summary;
    for (const auto & i : layers.issues) {
        report.diffs.push_back("layer_store: " + i);
    }

    const auto repeat = verify_materialization_repeatability(store, manifest, work_dir + "/repeat");
    report.materialization_repeatability = repeat.summary;

    const std::string mat_sha = sha256_file_hex(mat_path);
    report.extra.push_back({
            "file_sha256",
            (mat_sha == sha256_file_hex(original_gguf)) ? verify_status::ok : verify_status::fail,
            "full file SHA256 compare",
            { { "original", sha256_file_hex(original_gguf) }, { "materialized", mat_sha } },
    });

    const auto parity = verify_inference_parity(original_gguf, mat_path);
    report.logits   = parity.logits;
    report.sampling = parity.sampling;

    const bool ok =
            report.header.status == verify_status::ok &&
            report.metadata.status == verify_status::ok &&
            report.tensor_directory.status == verify_status::ok &&
            report.tensor_checksums.status == verify_status::ok &&
            report.alignment.status == verify_status::ok &&
            report.layer_store.status == verify_status::ok &&
            report.materialization_repeatability.status == verify_status::ok &&
            report.logits.status == verify_status::ok &&
            report.sampling.status == verify_status::ok;

    report.passed = ok;
    return report;
}
