#include "model_provider.h"

#include "../model_registry.h"

#include <memory>

using json = nlohmann::json;

// ---------------------------------------------------------------------------
// JSON serialization
// ---------------------------------------------------------------------------

json remote_model_file::to_json() const {
    return {
        { "filename",    filename },
        { "size_bytes",  size_bytes },
        { "etag",        etag },
        { "sha256",      sha256 },
        { "download_url", download_url },
    };
}

remote_model_file remote_model_file::from_json(const json & j) {
    remote_model_file f;
    f.filename    = j.value("filename", "");
    f.size_bytes  = j.value("size_bytes", static_cast<uint64_t>(0));
    f.etag        = j.value("etag", "");
    f.sha256      = j.value("sha256", "");
    f.download_url = j.value("download_url", "");
    return f;
}

json provider_discovery_result::to_json() const {
    json files_json = json::array();
    for (const auto & f : files) {
        files_json.push_back(f.to_json());
    }
    return {
        { "success",    success },
        { "provider",   provider },
        { "repository", repository },
        { "revision",   revision },
        { "files",      files_json },
        { "error",      error },
    };
}

provider_discovery_result provider_discovery_result::from_json(const json & j) {
    provider_discovery_result r;
    r.success    = j.value("success", false);
    r.provider   = j.value("provider", "");
    r.repository = j.value("repository", "");
    r.revision   = j.value("revision", "");
    r.error      = j.value("error", "");

    if (j.contains("files") && j["files"].is_array()) {
        for (const auto & item : j["files"]) {
            r.files.push_back(remote_model_file::from_json(item));
        }
    }
    return r;
}

// ---------------------------------------------------------------------------
// Factory
// ---------------------------------------------------------------------------

std::unique_ptr<model_provider> create_model_provider(const std::string & name) {
    if (name == "huggingface") {
        // Defined in huggingface_provider.cpp
        extern std::unique_ptr<model_provider> create_huggingface_provider();
        return create_huggingface_provider();
    }
    return nullptr;
}
