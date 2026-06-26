#include "gguf_diff.h"

#include "gguf_inspector.h"
#include "verification_common.h"

#include <set>

nlohmann::json gguf_diff_result::to_json() const {
    return {
        { "passed", passed },
        { "header", header.to_json() },
        { "metadata", metadata.to_json() },
        { "tensor_directory", tensor_directory.to_json() },
        { "differences", differences },
    };
}

static bool compare_meta_region(
        const std::string & a_path,
        const std::string & b_path,
        const size_t meta_size,
        std::vector<std::string> & diffs) {
    std::vector<uint8_t> a;
    std::vector<uint8_t> b;
    if (!read_file_range(a_path, 0, meta_size, a) || !read_file_range(b_path, 0, meta_size, b)) {
        diffs.push_back("failed to read metadata region");
        return false;
    }
    if (a == b) {
        return true;
    }
    for (size_t i = 0; i < a.size() && i < b.size(); ++i) {
        if (a[i] != b[i]) {
            diffs.push_back("metadata byte mismatch at offset " + std::to_string(i));
            if (diffs.size() >= 20) {
                diffs.push_back("... (truncated)");
                break;
            }
        }
    }
    if (a.size() != b.size()) {
        diffs.push_back("metadata size mismatch: " + std::to_string(a.size()) + " vs " + std::to_string(b.size()));
    }
    return false;
}

gguf_diff_result gguf_diff_files(const std::string & original, const std::string & materialized) {
    gguf_diff_result result;
    result.header.name            = "header";
    result.metadata.name          = "metadata";
    result.tensor_directory.name  = "tensor_directory";

    const auto orig = gguf_inspect_file(original);
    const auto mat  = gguf_inspect_file(materialized);

    if (orig.contains("error") || mat.contains("error")) {
        result.differences.push_back("failed to inspect one or both files");
        result.header.status = verify_status::fail;
        return result;
    }

    bool header_ok = true;
    for (const char * key : { "version", "alignment", "data_offset" }) {
        if (orig[key] != mat[key]) {
            header_ok = false;
            result.differences.push_back(std::string("header.") + key + ": " +
                    orig[key].dump() + " vs " + mat[key].dump());
        }
    }
    result.header.status  = header_ok ? verify_status::ok : verify_status::fail;
    result.header.message = header_ok ? "header fields match" : "header mismatch";

    const size_t meta_size = orig["metadata_size"].get<size_t>();
    const bool meta_ok = compare_meta_region(original, materialized, meta_size, result.differences);
    result.metadata.status  = meta_ok ? verify_status::ok : verify_status::fail;
    result.metadata.message = meta_ok ? "metadata region identical" : "metadata differs";

    std::map<std::string, nlohmann::json> orig_tensors;
    std::map<std::string, nlohmann::json> mat_tensors;
    for (const auto & t : orig["tensors"]) {
        orig_tensors[t["name"].get<std::string>()] = t;
    }
    for (const auto & t : mat["tensors"]) {
        mat_tensors[t["name"].get<std::string>()] = t;
    }

    bool dir_ok = true;
    std::set<std::string> names;
    for (const auto & kv : orig_tensors) {
        names.insert(kv.first);
    }
    for (const auto & kv : mat_tensors) {
        names.insert(kv.first);
    }

    for (const auto & name : names) {
        const bool in_o = orig_tensors.count(name) > 0;
        const bool in_m = mat_tensors.count(name) > 0;
        if (!in_o || !in_m) {
            dir_ok = false;
            result.differences.push_back("tensor missing: " + name + " (orig=" + std::to_string(in_o) +
                    " mat=" + std::to_string(in_m) + ")");
            continue;
        }
        const auto & ot = orig_tensors[name];
        const auto & mt = mat_tensors[name];
        for (const char * f : { "offset", "size_bytes", "ggml_type" }) {
            if (ot[f] != mt[f]) {
                dir_ok = false;
                result.differences.push_back("tensor " + name + " " + f + ": " + ot[f].dump() + " vs " + mt[f].dump());
            }
        }
        if (ot.contains("dims") && mt.contains("dims") && ot["dims"] != mt["dims"]) {
            dir_ok = false;
            result.differences.push_back("tensor " + name + " dims differ");
        }
    }

    result.tensor_directory.status  = dir_ok ? verify_status::ok : verify_status::fail;
    result.tensor_directory.message = dir_ok ? "tensor directory matches" : "tensor directory differs";
    result.passed = header_ok && meta_ok && dir_ok;
    return result;
}
