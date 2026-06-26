#include "tensor_verify.h"

#include "gguf_inspector.h"
#include "verification_common.h"

static tensor_verify_result tensor_verify_impl(
        const std::string & original,
        const std::string & materialized,
        const model_manifest * manifest,
        const worker_tensor_plan * plan) {
    tensor_verify_result result;
    result.summary.name = plan ? "tensor_checksums_worker" : "tensor_checksums";

    const auto orig = gguf_inspect_file(original);
    const auto mat  = gguf_inspect_file(materialized);
    if (!orig.contains("tensors") || !mat.contains("tensors")) {
        result.summary.status  = verify_status::fail;
        result.summary.message = "inspect failed";
        return result;
    }

    std::map<std::string, nlohmann::json> mat_by_name;
    for (const auto & t : mat["tensors"]) {
        mat_by_name[t["name"].get<std::string>()] = t;
    }

    std::map<std::string, tensor_descriptor> manifest_by_name;
    if (manifest) {
        for (const auto & td : manifest->tensors) {
            manifest_by_name[td.name] = td;
        }
    }

    int matched = 0;
    int total   = 0;

    for (const auto & ot : orig["tensors"]) {
        const std::string name = ot["name"].get<std::string>();

        if (plan && manifest) {
            const auto it = manifest_by_name.find(name);
            if (it == manifest_by_name.end()) {
                continue;
            }
            const auto & td = it->second;
            if (!gguf_tensor_included(
                        td,
                        plan->layer_start,
                        plan->layer_end,
                        plan->include_embedding,
                        plan->include_output,
                        manifest)) {
                result.skipped.push_back(name);
                continue;
            }
        }

        const auto it = mat_by_name.find(name);
        if (it == mat_by_name.end()) {
            result.mismatches.push_back("missing in materialized: " + name);
            continue;
        }

        const auto & mt = it->second;
        tensor_verify_entry entry;
        entry.name       = name;
        entry.offset     = ot["offset"].get<uint64_t>();
        entry.size_bytes = ot["size_bytes"].get<uint64_t>();
        entry.ggml_type  = ot["ggml_type"].get<std::string>();

        std::vector<uint8_t> a;
        std::vector<uint8_t> b;
        if (!read_file_range(original, entry.offset, entry.size_bytes, a) ||
                !read_file_range(materialized, mt["offset"].get<uint64_t>(), entry.size_bytes, b)) {
            result.mismatches.push_back("read failed: " + name);
            continue;
        }

        entry.sha256_original     = sha256_hex(a.data(), a.size());
        entry.sha256_materialized = sha256_hex(b.data(), b.size());
        entry.match               = (a == b);
        result.tensors.push_back(entry);
        ++total;
        if (entry.match) {
            ++matched;
        } else {
            result.mismatches.push_back("checksum mismatch: " + name);
        }
    }

    result.summary.status  = result.mismatches.empty() ? verify_status::ok : verify_status::fail;
    if (plan) {
        result.summary.message = std::to_string(matched) + "/" + std::to_string(total) +
                " included tensors match (" + worker_verify_role_to_string(plan->role) + ")";
        result.summary.details = {
            { "role", worker_verify_role_to_string(plan->role) },
            { "layer_start", plan->layer_start },
            { "layer_end", plan->layer_end },
            { "include_embedding", plan->include_embedding },
            { "include_output", plan->include_output },
            { "matched", matched },
            { "total", total },
            { "skipped", result.skipped.size() },
        };
    } else {
        result.summary.message = std::to_string(matched) + "/" + std::to_string(total) + " tensors match";
        result.summary.details = {
            { "matched", matched },
            { "total", total },
        };
    }
    return result;
}

tensor_verify_result tensor_verify_files(
        const std::string & original,
        const std::string & materialized) {
    return tensor_verify_impl(original, materialized, nullptr, nullptr);
}

tensor_verify_result tensor_verify_worker_files(
        const std::string & original,
        const std::string & materialized,
        const model_manifest & manifest,
        const worker_tensor_plan & plan) {
    return tensor_verify_impl(original, materialized, &manifest, &plan);
}
