#include "manifest_verify.h"

#include "gguf_inspector.h"

#include <map>
#include <set>

manifest_verify_result manifest_verify_against_gguf(
        const model_manifest & manifest,
        const std::string & gguf_path) {
    manifest_verify_result result;
    result.summary.name = "stage_1_manifest";

    const auto info = gguf_inspect_file(gguf_path);
    if (info.contains("error")) {
        result.summary.status  = verify_status::fail;
        result.summary.message = info["error"].get<std::string>();
        return result;
    }

    if (!info.contains("tensors")) {
        result.summary.status  = verify_status::fail;
        result.summary.message = "GGUF inspect missing tensors";
        return result;
    }

    const auto gguf_tensors = info["tensors"];
    if (gguf_tensors.size() != manifest.tensors.size()) {
        result.issues.push_back("tensor count mismatch: manifest=" +
                std::to_string(manifest.tensors.size()) +
                " gguf=" + std::to_string(gguf_tensors.size()));
    }

    std::map<std::string, nlohmann::json> by_name;
    for (const auto & t : gguf_tensors) {
        by_name[t["name"].get<std::string>()] = t;
    }

    uint64_t prev_offset = 0;
    bool offsets_ok      = true;

    for (const auto & mt : manifest.tensors) {
        const auto it = by_name.find(mt.name);
        if (it == by_name.end()) {
            result.issues.push_back("missing in GGUF: " + mt.name);
            continue;
        }

        const auto & gt = it->second;
        if (gt["offset"].get<uint64_t>() != mt.offset) {
            result.issues.push_back("offset mismatch: " + mt.name);
        }
        if (gt["size_bytes"].get<uint64_t>() != mt.size_bytes) {
            result.issues.push_back("size mismatch: " + mt.name);
        }
        if (gt["ggml_type"].get<std::string>() != mt.ggml_type) {
            result.issues.push_back("type mismatch: " + mt.name);
        }

        if (mt.offset < prev_offset) {
            offsets_ok = false;
        }
        prev_offset = std::max(prev_offset, mt.offset + mt.size_bytes);
    }

    if (!offsets_ok) {
        result.issues.push_back("tensor offsets do not increase monotonically");
    }

    for (const auto & [key, name] : manifest.special_tensors) {
        if (!name.empty() && by_name.find(name) == by_name.end()) {
            result.issues.push_back("special tensor missing: " + key + " (" + name + ")");
        }
    }

    result.summary.status  = result.issues.empty() ? verify_status::ok : verify_status::fail;
    result.summary.message = result.issues.empty()
            ? "manifest matches GGUF (" + std::to_string(manifest.tensors.size()) + " tensors)"
            : std::to_string(result.issues.size()) + " manifest issues";
    result.summary.details = {
        { "tensor_count", manifest.tensors.size() },
        { "issues", result.issues },
    };
    return result;
}
