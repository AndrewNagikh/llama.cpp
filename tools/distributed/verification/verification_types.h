#pragma once

#include "nlohmann/json.hpp"

#include <string>
#include <vector>

enum class verify_status {
    ok,
    fail,
    skip,
};

inline std::string verify_status_to_string(verify_status s) {
    switch (s) {
        case verify_status::ok:   return "OK";
        case verify_status::fail: return "FAIL";
        case verify_status::skip: return "SKIP";
    }
    return "UNKNOWN";
}

struct verify_check_result {
    std::string name;
    verify_status status = verify_status::skip;
    std::string message;
    nlohmann::json details = nlohmann::json::object();

    nlohmann::json to_json() const {
        return {
            { "name", name },
            { "status", verify_status_to_string(status) },
            { "message", message },
            { "details", details },
        };
    }
};

struct verification_report {
    std::string model_id;
    std::string original_path;
    std::string materialized_path;
    bool passed = false;

    verify_check_result header;
    verify_check_result metadata;
    verify_check_result tensor_directory;
    verify_check_result tensor_checksums;
    verify_check_result alignment;
    verify_check_result layer_store;
    verify_check_result materialization_repeatability;
    verify_check_result logits;
    verify_check_result sampling;

    std::vector<verify_check_result> extra;
    std::vector<std::string> diffs;

    nlohmann::json summary_json() const {
        auto field = [](const verify_check_result & c) {
            return verify_status_to_string(c.status);
        };
        return {
            { "model_id", model_id },
            { "passed", passed },
            { "header", field(header) },
            { "metadata", field(metadata) },
            { "tensor_directory", field(tensor_directory) },
            { "tensor_checksums", field(tensor_checksums) },
            { "alignment", field(alignment) },
            { "layer_store", field(layer_store) },
            { "materialization_repeatability", field(materialization_repeatability) },
            { "logits", field(logits) },
            { "sampling", field(sampling) },
        };
    }

    nlohmann::json to_json() const {
        nlohmann::json j = summary_json();
        j["original_path"]     = original_path;
        j["materialized_path"] = materialized_path;
        j["checks"] = nlohmann::json::array({
            header.to_json(),
            metadata.to_json(),
            tensor_directory.to_json(),
            tensor_checksums.to_json(),
            alignment.to_json(),
            layer_store.to_json(),
            materialization_repeatability.to_json(),
            logits.to_json(),
            sampling.to_json(),
        });
        for (const auto & e : extra) {
            j["checks"].push_back(e.to_json());
        }
        j["diffs"] = diffs;
        return j;
    }
};
