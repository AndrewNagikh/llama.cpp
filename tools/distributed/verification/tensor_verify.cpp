#include "tensor_verify.h"

#include "gguf_inspector.h"
#include "verification_common.h"

tensor_verify_result tensor_verify_files(
        const std::string & original,
        const std::string & materialized) {
    tensor_verify_result result;
    result.summary.name = "tensor_checksums";

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

    int matched = 0;
    int total   = 0;

    for (const auto & ot : orig["tensors"]) {
        const std::string name = ot["name"].get<std::string>();
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
    result.summary.message = std::to_string(matched) + "/" + std::to_string(total) + " tensors match";
    result.summary.details = {
        { "matched", matched },
        { "total", total },
    };
    return result;
}
