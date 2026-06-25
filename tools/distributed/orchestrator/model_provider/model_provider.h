#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "nlohmann/json.hpp"

// Forward declaration: the provider works with registry records but does not
// need to know the full registry implementation.
struct dist_model_record;

// ---------------------------------------------------------------------------
// Remote model discovery – Task 9.2
//
// A Model Provider answers the question: "what files are available for this
// model in a remote repository?"  It never downloads GGUF files, never parses
// GGUF metadata, and never uses HTTP Range requests.
// ---------------------------------------------------------------------------

struct remote_model_file {
    std::string filename;
    uint64_t    size_bytes = 0;
    std::string etag;
    std::string sha256;
    std::string download_url;

    nlohmann::json to_json() const;
    static remote_model_file from_json(const nlohmann::json & j);
};

struct provider_discovery_result {
    bool        success    = false;
    std::string provider;
    std::string repository;
    std::string revision;
    std::vector<remote_model_file> files;
    std::string error;

    nlohmann::json to_json() const;
    static provider_discovery_result from_json(const nlohmann::json & j);
};

class model_provider {
public:
    virtual ~model_provider() = default;

    virtual std::string name() const = 0;

    virtual provider_discovery_result discover(const dist_model_record & model) = 0;
};

// Factory. Returns nullptr for unknown provider names.
std::unique_ptr<model_provider> create_model_provider(const std::string & name);
