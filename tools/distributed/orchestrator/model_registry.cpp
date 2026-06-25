#include "model_registry.h"

#include <algorithm>
#include <cctype>
#include <chrono>
#include <ctime>
#include <iomanip>
#include <sstream>
#include <stdexcept>

using json = nlohmann::json;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

static std::string format_time(const std::chrono::system_clock::time_point & tp) {
    const std::time_t t = std::chrono::system_clock::to_time_t(tp);
    std::tm tm{};
#if defined(_WIN32)
    gmtime_s(&tm, &t);
#else
    gmtime_r(&t, &tm);
#endif
    std::ostringstream oss;
    oss << std::put_time(&tm, "%Y-%m-%dT%H:%M:%SZ");
    return oss.str();
}

// ---------------------------------------------------------------------------
// Model status
// ---------------------------------------------------------------------------

std::string dist_model_status_to_string(dist_model_status s) {
    switch (s) {
        case dist_model_status::discovered:           return "DISCOVERED";
        case dist_model_status::manifest_pending:    return "MANIFEST_PENDING";
        case dist_model_status::manifest_ready:      return "MANIFEST_READY";
        case dist_model_status::installing:          return "INSTALLING";
        case dist_model_status::partially_available: return "PARTIALLY_AVAILABLE";
        case dist_model_status::available:           return "AVAILABLE";
        case dist_model_status::degraded:            return "DEGRADED";
        case dist_model_status::unavailable:         return "UNAVAILABLE";
    }
    return "DISCOVERED";
}

dist_model_status dist_model_status_from_string(const std::string & s) {
    std::string upper;
    upper.reserve(s.size());
    for (char c : s) {
        upper += static_cast<char>(std::toupper(static_cast<unsigned char>(c)));
    }

    if (upper == "DISCOVERED")           return dist_model_status::discovered;
    if (upper == "MANIFEST_PENDING")    return dist_model_status::manifest_pending;
    if (upper == "MANIFEST_READY")      return dist_model_status::manifest_ready;
    if (upper == "INSTALLING")          return dist_model_status::installing;
    if (upper == "PARTIALLY_AVAILABLE") return dist_model_status::partially_available;
    if (upper == "AVAILABLE")           return dist_model_status::available;
    if (upper == "DEGRADED")            return dist_model_status::degraded;
    if (upper == "UNAVAILABLE")         return dist_model_status::unavailable;

    return dist_model_status::discovered;
}

// ---------------------------------------------------------------------------
// Manifest
// ---------------------------------------------------------------------------

json model_manifest::to_json() const {
    return json::object();
}

model_manifest model_manifest::from_json(const json & /*j*/) {
    // Task 9.1: manifest is intentionally empty. Parsing always succeeds.
    return model_manifest{};
}

// ---------------------------------------------------------------------------
// Record
// ---------------------------------------------------------------------------

json dist_model_record::to_json() const {
    json files_json = json::array();
    for (const auto & f : files) {
        files_json.push_back(f.to_json());
    }

    json j = {
        { "model_id",    model_id },
        { "display_name", display_name },
        { "provider",     source },
        { "source",       source },
        { "repository",   repository },
        { "filename",     filename },
        { "revision",     revision },
        { "architecture", architecture },
        { "status",       dist_model_status_to_string(status) },
        { "files",        files_json },
        { "provider_revision", provider_revision },
        { "provider_etag",     provider_etag },
    };

    if (last_discovery.time_since_epoch().count() > 0) {
        j["last_discovery"] = format_time(last_discovery);
    } else {
        j["last_discovery"] = nullptr;
    }

    if (manifest.has_value()) {
        j["manifest"] = manifest->to_json();
    } else {
        j["manifest"] = nullptr;
    }

    return j;
}

dist_model_record dist_model_record_from_json(const json & j) {
    dist_model_record r;
    r.model_id     = j.value("model_id", "");
    r.display_name = j.value("display_name", "");
    r.source       = j.value("source", "");
    r.repository   = j.value("repository", "");
    r.filename     = j.value("filename", "");
    r.revision     = j.value("revision", "");
    r.architecture = j.value("architecture", "");

    // Task 9.1 forces every new registration to DISCOVERED.
    r.status = dist_model_status::discovered;

    if (j.contains("manifest") && j["manifest"].is_object()) {
        r.manifest = model_manifest::from_json(j["manifest"]);
    } else {
        r.manifest = std::nullopt;
    }

    if (j.contains("files") && j["files"].is_array()) {
        r.files.clear();
        for (const auto & item : j["files"]) {
            r.files.push_back(remote_model_file::from_json(item));
        }
    }
    r.provider_revision = j.value("provider_revision", "");
    r.provider_etag     = j.value("provider_etag", "");

    return r;
}

// ---------------------------------------------------------------------------
// Registry
// ---------------------------------------------------------------------------

void cluster_model_registry::add_or_update(const dist_model_record & record) {
    std::lock_guard<std::mutex> lock(mutex_);
    records_[record.model_id] = record;
}

bool cluster_model_registry::remove(const std::string & model_id) {
    std::lock_guard<std::mutex> lock(mutex_);
    const auto it = records_.find(model_id);
    if (it == records_.end()) {
        return false;
    }
    records_.erase(it);
    return true;
}

const dist_model_record * cluster_model_registry::find(const std::string & model_id) const {
    std::lock_guard<std::mutex> lock(mutex_);
    const auto it = records_.find(model_id);
    if (it == records_.end()) {
        return nullptr;
    }
    return &it->second;
}

dist_model_record * cluster_model_registry::find(const std::string & model_id) {
    std::lock_guard<std::mutex> lock(mutex_);
    const auto it = records_.find(model_id);
    if (it == records_.end()) {
        return nullptr;
    }
    return &it->second;
}

std::vector<dist_model_record> cluster_model_registry::list() const {
    std::lock_guard<std::mutex> lock(mutex_);
    std::vector<dist_model_record> result;
    result.reserve(records_.size());
    for (const auto & kv : records_) {
        result.push_back(kv.second);
    }
    return result;
}

bool cluster_model_registry::apply_discovery(
        const std::string & model_id,
        const provider_discovery_result & result,
        dist_model_record * out) {
    std::lock_guard<std::mutex> lock(mutex_);
    const auto it = records_.find(model_id);
    if (it == records_.end()) {
        return false;
    }

    dist_model_record & r = it->second;
    r.files             = result.files;
    r.provider_revision = result.revision;
    r.provider_etag     = result.revision; // version marker
    r.last_discovery    = std::chrono::system_clock::now();
    r.status            = dist_model_status::manifest_pending;

    if (out) {
        *out = r;
    }
    return true;
}
