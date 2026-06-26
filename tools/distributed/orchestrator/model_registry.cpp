#include "model_registry.h"

#include <algorithm>
#include <cctype>
#include <chrono>
#include <ctime>
#include <iomanip>
#include <set>
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

    if (layout.has_value()) {
        j["layout"] = layout->to_json();
    } else {
        j["layout"] = nullptr;
    }

    if (actual.has_value()) {
        j["actual"] = actual->to_json();
    } else {
        j["actual"] = nullptr;
    }

    if (coverage.has_value()) {
        j["coverage"] = coverage->to_json();
    } else {
        j["coverage"] = nullptr;
    }

    if (stored_install_plan.has_value()) {
        j["install_plan"] = stored_install_plan->to_json();
    } else {
        j["install_plan"] = nullptr;
    }

    if (pending_layout.has_value()) {
        j["pending_layout"] = pending_layout->to_json();
    } else {
        j["pending_layout"] = nullptr;
    }

    if (optimization.has_value()) {
        j["optimization"] = optimization->to_json();
    } else {
        j["optimization"] = nullptr;
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

    if (j.contains("layout") && j["layout"].is_object()) {
        r.layout = model_layout::from_json(j["layout"]);
    } else {
        r.layout = std::nullopt;
    }

    if (j.contains("actual") && j["actual"].is_object()) {
        r.actual = actual_model_layout::from_json(j["actual"]);
    } else {
        r.actual = std::nullopt;
    }

    if (j.contains("coverage") && j["coverage"].is_object()) {
        r.coverage = coverage_report::from_json(j["coverage"]);
    } else {
        r.coverage = std::nullopt;
    }

    if (j.contains("install_plan") && j["install_plan"].is_object()) {
        r.stored_install_plan = install_plan::from_json(j["install_plan"]);
    } else {
        r.stored_install_plan = std::nullopt;
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

bool cluster_model_registry::apply_manifest(
        const std::string & model_id,
        const model_manifest & manifest,
        dist_model_record * out) {
    std::lock_guard<std::mutex> lock(mutex_);
    const auto it = records_.find(model_id);
    if (it == records_.end()) {
        return false;
    }

    dist_model_record & r = it->second;
    r.manifest     = manifest;
    r.architecture = manifest.architecture;
    r.status       = dist_model_status::manifest_ready;

    if (out) {
        *out = r;
    }
    return true;
}

bool cluster_model_registry::apply_layout(
        const std::string & model_id,
        const desired_model_layout & layout,
        dist_model_record * out) {
    std::lock_guard<std::mutex> lock(mutex_);
    const auto it = records_.find(model_id);
    if (it == records_.end()) {
        return false;
    }

    dist_model_record & r = it->second;
    model_layout ml;
    ml.desired = layout;
    r.layout = ml;

    if (out) {
        *out = r;
    }
    return true;
}

bool cluster_model_registry::apply_actual(
        const std::string & model_id,
        const actual_model_layout & actual_layout,
        dist_model_record * out) {
    std::lock_guard<std::mutex> lock(mutex_);
    const auto it = records_.find(model_id);
    if (it == records_.end()) {
        return false;
    }

    dist_model_record & r = it->second;
    r.actual = actual_layout;

    if (out) {
        *out = r;
    }
    return true;
}

bool cluster_model_registry::apply_coverage(
        const std::string & model_id,
        const coverage_report & report,
        dist_model_record * out) {
    std::lock_guard<std::mutex> lock(mutex_);
    const auto it = records_.find(model_id);
    if (it == records_.end()) {
        return false;
    }

    dist_model_record & r = it->second;
    r.coverage = report;

    if (out) {
        *out = r;
    }
    return true;
}

bool cluster_model_registry::refresh_coverage(
        const std::string & model_id,
        const std::set<std::string> & online_nodes,
        dist_model_record * out) {
    std::lock_guard<std::mutex> lock(mutex_);
    const auto it = records_.find(model_id);
    if (it == records_.end()) {
        return false;
    }

    dist_model_record & r = it->second;
    if (!r.layout.has_value()) {
        return false;
    }

    actual_model_layout actual{};
    actual.model_id = model_id;
    if (r.actual.has_value()) {
        actual = *r.actual;
    }

    const coverage_report report = compute_coverage(
            r.layout->desired, actual, online_nodes);
    r.coverage = report;

    if (out) {
        *out = r;
    }
    return true;
}

bool cluster_model_registry::apply_install_plan(
        const std::string & model_id,
        const install_plan & plan,
        dist_model_record * out) {
    std::lock_guard<std::mutex> lock(mutex_);
    const auto it = records_.find(model_id);
    if (it == records_.end()) {
        return false;
    }

    dist_model_record & r = it->second;
    r.stored_install_plan = plan;

    if (out) {
        *out = r;
    }
    return true;
}

bool cluster_model_registry::apply_optimization(
        const std::string & model_id,
        const optimization_result & result,
        dist_model_record * out) {
    std::lock_guard<std::mutex> lock(mutex_);
    const auto it = records_.find(model_id);
    if (it == records_.end()) {
        return false;
    }

    dist_model_record & r = it->second;
    r.optimization = result;

    if (out) {
        *out = r;
    }
    return true;
}

bool cluster_model_registry::apply_pending_layout(
        const std::string & model_id,
        const desired_model_layout & layout,
        dist_model_record * out) {
    std::lock_guard<std::mutex> lock(mutex_);
    const auto it = records_.find(model_id);
    if (it == records_.end()) {
        return false;
    }

    model_layout ml;
    ml.desired = layout;
    it->second.pending_layout = ml;

    if (out) {
        *out = it->second;
    }
    return true;
}

bool cluster_model_registry::commit_pending_layout(
        const std::string & model_id,
        dist_model_record * out) {
    std::lock_guard<std::mutex> lock(mutex_);
    const auto it = records_.find(model_id);
    if (it == records_.end() || !it->second.pending_layout.has_value()) {
        return false;
    }

    it->second.layout = *it->second.pending_layout;
    it->second.pending_layout = std::nullopt;

    if (out) {
        *out = it->second;
    }
    return true;
}

bool cluster_model_registry::discard_pending_layout(
        const std::string & model_id,
        dist_model_record * out) {
    std::lock_guard<std::mutex> lock(mutex_);
    const auto it = records_.find(model_id);
    if (it == records_.end()) {
        return false;
    }

    it->second.pending_layout = std::nullopt;

    if (out) {
        *out = it->second;
    }
    return true;
}

bool cluster_model_registry::refresh_pending_coverage(
        const std::string & model_id,
        const std::set<std::string> & online_nodes,
        dist_model_record * out) {
    std::lock_guard<std::mutex> lock(mutex_);
    const auto it = records_.find(model_id);
    if (it == records_.end() || !it->second.pending_layout.has_value()) {
        return false;
    }

    actual_model_layout actual{};
    actual.model_id = model_id;
    if (it->second.actual.has_value()) {
        actual = *it->second.actual;
    }

    const coverage_report report = compute_coverage(
            it->second.pending_layout->desired, actual, online_nodes);
    it->second.coverage = report;

    if (out) {
        *out = it->second;
    }
    return true;
}
