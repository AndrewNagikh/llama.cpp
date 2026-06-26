#include "worker_tensor_plan.h"

#include "architecture/architecture_descriptor.h"
#include "node_agent/layer_store/descriptor_materialize.h"
#include "architecture/semantic_blob.h"
#include "architecture/tensor_plan.h"
#include "gguf_inspector.h"

#include <algorithm>
#include <map>
#include <set>

std::string worker_verify_role_to_string(const worker_verify_role role) {
    switch (role) {
        case worker_verify_role::entry:  return "ENTRY";
        case worker_verify_role::middle: return "MIDDLE";
        case worker_verify_role::final:  return "FINAL";
        case worker_verify_role::full:   return "FULL";
    }
    return "UNKNOWN";
}

bool gguf_tensor_included(
        const tensor_descriptor & t,
        const int32_t layer_start,
        const int32_t layer_end,
        const bool include_embedding,
        const bool include_output,
        const model_manifest * manifest) {
    if (manifest == nullptr) {
        if (t.role == tensor_role::embedding && include_embedding) {
            return true;
        }
        if (include_embedding && t.layer < layer_start &&
                t.role != tensor_role::output_norm &&
                t.role != tensor_role::lm_head) {
            return true;
        }
        if (include_output &&
                (t.role == tensor_role::output_norm || t.role == tensor_role::lm_head)) {
            return true;
        }
        return t.layer >= layer_start && t.layer < layer_end;
    }

    const auto desc = build_architecture_descriptor(*manifest);
    return descriptor_tensor_included(
            desc, t, layer_start, layer_end, include_embedding, include_output);
}

std::vector<std::string> list_tensors_for_worker(
        const model_manifest & manifest,
        const worker_tensor_plan & plan) {
    std::vector<std::string> names;
    for (const auto & t : manifest.tensors) {
        if (gguf_tensor_included(
                    t,
                    plan.layer_start,
                    plan.layer_end,
                    plan.include_embedding,
                    plan.include_output,
                    &manifest)) {
            names.push_back(t.name);
        }
    }
    return names;
}

materialized_special_flags inspect_materialized_special_tensors(
        const std::string & gguf_path,
        const model_manifest & manifest) {
    materialized_special_flags flags;
    const auto info = gguf_inspect_file(gguf_path);
    if (!info.contains("tensors")) {
        return flags;
    }

    std::set<std::string> names;
    for (const auto & t : info["tensors"]) {
        names.insert(t["name"].get<std::string>());
    }

    auto has_tensor = [&](const std::string & key) -> bool {
        const auto it = manifest.special_tensors.find(key);
        if (it != manifest.special_tensors.end() && !it->second.empty()) {
            return names.count(it->second) > 0;
        }
        return false;
    };

    if (!manifest.empty()) {
        const auto desc = build_architecture_descriptor(manifest);
        flags.embedding   = has_tensor("embedding");
        flags.output_norm = has_tensor("output_norm");
        if (desc.separate_lm_head) {
            flags.output = has_tensor("lm_head");
        } else {
            flags.output = flags.embedding;
        }
        return flags;
    }

    flags.embedding   = names.count("token_embd.weight") > 0;
    flags.output_norm = names.count("output_norm.weight") > 0;
    flags.output      = names.count("output.weight") > 0 || flags.embedding;
    return flags;
}

static bool special_flags_match_role(
        const materialized_special_flags & flags,
        const worker_tensor_plan & plan,
        const model_manifest & manifest) {
    const auto desc = build_architecture_descriptor(manifest);
    if (plan.include_embedding && !flags.embedding) {
        return false;
    }
    if (plan.include_output) {
        if (find_blob(desc.blobs, "output_norm") && !flags.output_norm) {
            return false;
        }
        if (desc.separate_lm_head && !flags.output) {
            return false;
        }
        if (desc.tied_embeddings && !flags.embedding) {
            return false;
        }
    }
    return true;
}

verify_check_result verify_worker_tensor_plan(
        const model_manifest & manifest,
        const worker_tensor_plan & plan) {
    verify_check_result result;
    result.name = "stage_3_materializer_input";

    const auto expected = list_tensors_for_worker(manifest, plan);
    if (expected.empty()) {
        result.status  = verify_status::fail;
        result.message = "no tensors in worker plan";
        return result;
    }

    std::vector<std::string> missing;
    for (const auto & t : manifest.tensors) {
        const bool required = gguf_tensor_included(
                t,
                plan.layer_start,
                plan.layer_end,
                plan.include_embedding,
                plan.include_output,
                &manifest);
        if (!required) {
            continue;
        }
        const auto it = std::find(expected.begin(), expected.end(), t.name);
        if (it == expected.end()) {
            missing.push_back(t.name);
        }
    }

    result.details = {
        { "role", worker_verify_role_to_string(plan.role) },
        { "layer_start", plan.layer_start },
        { "layer_end", plan.layer_end },
        { "include_embedding", plan.include_embedding },
        { "include_output", plan.include_output },
        { "tensor_count", expected.size() },
        { "tensors", expected },
    };

    if (!missing.empty()) {
        result.status          = verify_status::fail;
        result.message         = "required tensors missing from plan";
        result.details["missing"] = missing;
        return result;
    }

    result.status  = verify_status::ok;
    result.message = worker_verify_role_to_string(plan.role) + " plan has " +
            std::to_string(expected.size()) + " tensors";
    return result;
}

verify_check_result verify_materialized_structure_for_role(
        const std::string & gguf_path,
        const worker_tensor_plan & plan,
        const model_manifest & manifest) {
    verify_check_result result;
    result.name = "stage_4_materialized_structure";

    const auto info = gguf_inspect_file(gguf_path);
    if (info.contains("error")) {
        result.status  = verify_status::fail;
        result.message = info["error"].get<std::string>();
        return result;
    }

    const auto flags = inspect_materialized_special_tensors(gguf_path, manifest);
    result.details = {
        { "role", worker_verify_role_to_string(plan.role) },
        { "version", info.value("version", 0) },
        { "alignment", info.value("alignment", 0) },
        { "metadata_size", info.value("metadata_size", 0) },
        { "tensor_count", info.value("tensor_count", 0) },
        { "special", {
            { "embedding", flags.embedding },
            { "output", flags.output },
            { "output_norm", flags.output_norm },
        } },
    };

    if (!special_flags_match_role(flags, plan, manifest)) {
        result.status  = verify_status::fail;
        result.message = "missing required special tensors for " +
                worker_verify_role_to_string(plan.role);
        return result;
    }

    result.status  = verify_status::ok;
    result.message = "materialized GGUF structure OK for " +
            worker_verify_role_to_string(plan.role);
    return result;
}

verify_check_result verify_worker_store_assignment(
        const layer_store & store,
        const model_manifest & manifest,
        const worker_tensor_plan & plan) {
    verify_check_result result;
    result.name = "stage_7_worker_assignment";

    const auto desc = build_architecture_descriptor(manifest);
    std::vector<std::string> issues;

    std::vector<std::string> required_blobs;
    if (plan.include_embedding) {
        required_blobs.push_back("embedding");
    }
    if (plan.include_output) {
        if (find_blob(desc.blobs, "output_norm")) {
            required_blobs.push_back("output_norm");
        }
        if (find_blob(desc.blobs, "output_head")) {
            required_blobs.push_back("output_head");
        }
    }

    std::string blob_err;
    if (!verify_required_blobs(store, desc, required_blobs, blob_err)) {
        issues.push_back(blob_err);
    }

    for (int32_t layer = plan.layer_start; layer < plan.layer_end; ++layer) {
        if (!store.has_layer(layer)) {
            issues.push_back("missing layer " + std::to_string(layer));
        }
    }

    std::vector<std::string> special;
    for (const std::string & blob_id : required_blobs) {
        const semantic_blob * blob = find_blob(desc.blobs, blob_id);
        if (blob == nullptr) {
            continue;
        }
        for (const auto & slot : blob->tensors) {
            special.push_back(slot.name);
        }
    }

    result.details = {
        { "role", worker_verify_role_to_string(plan.role) },
        { "layers", { plan.layer_start, plan.layer_end } },
        { "special_tensors", special },
        { "issues", issues },
    };

    if (!issues.empty()) {
        result.status  = verify_status::fail;
        result.message = "worker store assignment incomplete";
        return result;
    }

    result.status  = verify_status::ok;
    result.message = "worker store assignment OK";
    return result;
}
