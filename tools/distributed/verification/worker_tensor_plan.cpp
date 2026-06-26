#include "worker_tensor_plan.h"

#include "gguf_inspector.h"
#include "node_agent/layer_store/layer_special.h"

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

worker_tensor_plan make_worker_tensor_plan(
        const worker_verify_role role,
        const int32_t layer_start,
        const int32_t layer_end) {
    worker_tensor_plan plan;
    plan.role         = role;
    plan.layer_start  = layer_start;
    plan.layer_end    = layer_end;
    switch (role) {
        case worker_verify_role::entry:
            plan.include_embedding = true;
            plan.include_output    = false;
            break;
        case worker_verify_role::middle:
            plan.include_embedding = false;
            plan.include_output    = false;
            break;
        case worker_verify_role::final:
            plan.include_embedding = false;
            plan.include_output    = true;
            break;
        case worker_verify_role::full:
            plan.include_embedding = true;
            plan.include_output    = true;
            break;
    }
    return plan;
}

bool gguf_tensor_included(
        const tensor_descriptor & t,
        const int32_t layer_start,
        const int32_t layer_end,
        const bool include_embedding,
        const bool include_output,
        const model_manifest * manifest) {
    if (t.role == tensor_role::embedding && include_embedding) {
        return true;
    }
    // Tied lm_head models reuse token_embd.weight at the final stage.
    if (include_output && manifest != nullptr &&
            manifest->special_tensors.count("lm_head") == 0 &&
            t.role == tensor_role::embedding) {
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
        flags.embedding   = has_tensor("embedding");
        flags.output_norm = has_tensor("output_norm");
        if (manifest.special_tensors.count("lm_head") > 0) {
            flags.output = has_tensor("lm_head");
        } else {
            // Tied embeddings: lm_head shares token_embd.weight.
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
    if (plan.include_embedding && !flags.embedding) {
        return false;
    }
    if (plan.include_output) {
        if (manifest.special_tensors.count("output_norm") > 0 && !flags.output_norm) {
            return false;
        }
        if (manifest.special_tensors.count("lm_head") > 0 && !flags.output) {
            return false;
        }
        if (manifest.special_tensors.count("lm_head") == 0 && !flags.embedding) {
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

    if (plan.role == worker_verify_role::full) {
        if (!special_flags_match_role(flags, plan, manifest)) {
            result.status  = verify_status::fail;
            result.message = "full GGUF missing required special tensors";
            return result;
        }
    } else if (!special_flags_match_role(flags, plan, manifest)) {
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

    std::vector<std::string> issues;
    if (plan.include_embedding && !store.has_layer(layer_special::embedding)) {
        issues.push_back("missing embedding blob (-1)");
    }
    for (int32_t layer = plan.layer_start; layer < plan.layer_end; ++layer) {
        if (!store.has_layer(layer)) {
            issues.push_back("missing layer " + std::to_string(layer));
        }
    }
    if (plan.include_output) {
        if (manifest.special_tensors.count("lm_head") == 0 && !store.has_layer(layer_special::embedding)) {
            issues.push_back("missing tied lm_head embedding blob (-1)");
        }
        if (!store.has_layer(layer_special::output)) {
            issues.push_back("missing output blob (-2)");
        }
    }

    std::vector<std::string> special;
    if (plan.include_embedding ||
            (plan.include_output && manifest.special_tensors.count("lm_head") == 0)) {
        special.push_back("token_embd.weight");
    }
    if (plan.include_output) {
        if (manifest.special_tensors.count("output_norm") > 0) {
            special.push_back(manifest.special_tensors.at("output_norm"));
        }
        if (manifest.special_tensors.count("lm_head") > 0) {
            special.push_back(manifest.special_tensors.at("lm_head"));
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
        result.message = issues.front();
        return result;
    }

    result.status  = verify_status::ok;
    result.message = worker_verify_role_to_string(plan.role) + " store assignment OK";
    return result;
}
