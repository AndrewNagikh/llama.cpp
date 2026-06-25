#pragma once

#include <chrono>
#include <map>
#include <mutex>
#include <optional>
#include <string>
#include <vector>

#include "nlohmann/json.hpp"
#include "manifest_builder/manifest_builder.h"
#include "model_provider/model_provider.h"

// ---------------------------------------------------------------------------
// Cluster Model Registry - Task 9.1 / 9.2 / 9.3
//
// The registry is the single source of truth about which models are known
// to the cluster. It stores metadata and GGUF manifests but never downloads
// or stores tensor weights.
// ---------------------------------------------------------------------------

enum class dist_model_status {
    discovered,
    manifest_pending,
    manifest_ready,
    installing,
    partially_available,
    available,
    degraded,
    unavailable
};

std::string dist_model_status_to_string(dist_model_status s);
dist_model_status dist_model_status_from_string(const std::string & s);

struct dist_model_record {
    std::string model_id;
    std::string display_name;
    std::string source;
    std::string repository;
    std::string filename;
    std::string revision;
    std::string architecture;
    dist_model_status status = dist_model_status::discovered;
    std::optional<model_manifest> manifest;

    // Task 9.2: remote discovery metadata
    std::vector<remote_model_file> files;
    std::string provider_revision;
    std::string provider_etag;
    std::chrono::system_clock::time_point last_discovery;

    nlohmann::json to_json() const;
};

// Parse a POST /models/register payload into a record.
// Missing fields are left empty; status is forced to DISCOVERED.
dist_model_record dist_model_record_from_json(const nlohmann::json & j);

class cluster_model_registry {
public:
    // Add a new model or update an existing one (matched by model_id).
    void add_or_update(const dist_model_record & record);

    // Remove a model record. Returns true if it existed.
    bool remove(const std::string & model_id);

    // Find one record. nullptr if not found.
    const dist_model_record * find(const std::string & model_id) const;
    dist_model_record * find(const std::string & model_id);

    // Return all registered models.
    std::vector<dist_model_record> list() const;

    // Merge a successful provider discovery result into an existing record.
    // Sets status to MANIFEST_PENDING and updates files / provider revision.
    // Returns false if the model_id is not known.
    bool apply_discovery(
            const std::string & model_id,
            const provider_discovery_result & result,
            dist_model_record * out = nullptr);

    // Merge a built manifest into an existing record.
    // Sets status to MANIFEST_READY and updates architecture.
    // Returns false if the model_id is not known.
    bool apply_manifest(
            const std::string & model_id,
            const model_manifest & manifest,
            dist_model_record * out = nullptr);

private:
    mutable std::mutex mutex_;
    std::map<std::string, dist_model_record> records_;
};
