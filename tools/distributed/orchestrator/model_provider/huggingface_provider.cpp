#include "huggingface_provider.h"

#include "../model_registry.h"

#include "httplib.h"
#include "nlohmann/json.hpp"

#include <cstdlib>
#include <string>

using json = nlohmann::json;

namespace {

static std::string hf_token() {
    const char * token = std::getenv("HF_TOKEN");
    return token ? std::string(token) : std::string{};
}

class huggingface_provider_impl : public model_provider {
public:
    std::string name() const override { return "huggingface"; }

    provider_discovery_result discover(const dist_model_record & model) override {
        provider_discovery_result result{};
        result.provider   = name();
        result.repository = model.repository;
        result.revision   = model.revision;

        if (model.repository.empty()) {
            result.error = "repository is empty";
            return result;
        }

        httplib::Client cli("https://huggingface.co");
        cli.set_connection_timeout(10, 0);
        cli.set_read_timeout(30, 0);
        cli.set_follow_location(true);

        httplib::Headers headers = {
            { "User-Agent", "distributed-llama-node-agent/0.1" },
            { "Accept", "application/json" }
        };

        const std::string token = hf_token();
        if (!token.empty()) {
            headers.emplace("Authorization", "Bearer " + token);
        }

        const std::string revision = model.revision.empty() ? "main" : model.revision;
        const std::string path = "/api/models/" + model.repository + "/tree/" + revision;

        const auto res = cli.Get(path.c_str(), headers);
        if (!res) {
            result.error = "network error: could not reach Hugging Face";
            return result;
        }

        if (res->status == 401 || res->status == 403) {
            result.error = "unauthorized: check HF_TOKEN or repository visibility";
            return result;
        }
        if (res->status == 404) {
            result.error = "repository or revision not found";
            return result;
        }
        if (res->status != 200) {
            result.error = "Hugging Face API returned HTTP " + std::to_string(res->status);
            return result;
        }

        json body;
        try {
            body = json::parse(res->body);
        } catch (...) {
            result.error = "invalid JSON response from Hugging Face";
            return result;
        }

        if (!body.is_array()) {
            result.error = "unexpected response shape from Hugging Face";
            return result;
        }

        for (const auto & item : body) {
            if (item.value("type", "") != "file") {
                continue;
            }

            remote_model_file file{};
            file.filename = item.value("path", "");
            file.size_bytes = item.value("size", static_cast<uint64_t>(0));
            file.sha256 = item.value("oid", "");
            file.etag   = file.sha256;
            file.download_url = "https://huggingface.co/" + model.repository +
                                "/resolve/" + revision + "/" + file.filename;

            if (!file.filename.empty()) {
                result.files.push_back(std::move(file));
            }
        }

        if (result.files.empty()) {
            result.error = "no files found in repository";
            return result;
        }

        result.success = true;
        return result;
    }
};

} // namespace

std::unique_ptr<model_provider> create_huggingface_provider() {
    return std::make_unique<huggingface_provider_impl>();
}
