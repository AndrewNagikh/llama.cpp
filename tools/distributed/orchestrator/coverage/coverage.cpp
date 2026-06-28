#include "coverage.h"

#include "architecture/semantic_blob.h"
#include "dist_common.h"

#include <algorithm>
#include <cctype>
#include <cstring>
#include <map>
#include <optional>
#include <set>
#include <tuple>

using json = nlohmann::json;

namespace {

static std::string upper_ascii(std::string s) {
    for (char & c : s) {
        c = static_cast<char>(std::toupper(static_cast<unsigned char>(c)));
    }
    return s;
}

static std::string layer_key(int32_t layer_index, const std::string & node_id) {
    return std::to_string(layer_index) + "@" + node_id;
}

static std::string actual_layer_key(const installed_layer & layer) {
    if (!layer.blob_id.empty() && !layer.tensor_name.empty()) {
        return layer.node_id + ":blob:" + layer.blob_id + ":" + layer.tensor_name;
    }
    return layer_key(layer.layer_index, layer.node_id);
}

static std::optional<int32_t> layer_index_from_blob_id(const std::string & blob_id) {
    constexpr const char * prefix = "layer:";
    if (blob_id.rfind(prefix, 0) != 0) {
        return std::nullopt;
    }
    try {
        return static_cast<int32_t>(std::stoi(blob_id.substr(std::strlen(prefix))));
    } catch (...) {
        return std::nullopt;
    }
}

static std::map<std::string, installed_layer> index_actual_layers(
        const actual_model_layout & actual) {
    std::map<std::string, installed_layer> indexed;
    for (const auto & layer : actual.layers) {
        if (layer.layer_index >= 0 && !layer.node_id.empty()) {
            indexed[layer_key(layer.layer_index, layer.node_id)] = layer;
        }
    }

    struct layer_blob_agg {
        int32_t     layer_index = -1;
        std::string node_id;
        int         ready_count = 0;
        int         total_count = 0;
        bool        any_corrupted = false;
        uint64_t    size_bytes = 0;
        std::string device;
    };

    std::map<std::string, layer_blob_agg> blob_layers;
    for (const auto & layer : actual.layers) {
        if (layer.blob_id.empty() || layer.tensor_name.empty() || layer.node_id.empty()) {
            continue;
        }
        const std::optional<int32_t> parsed = layer_index_from_blob_id(layer.blob_id);
        if (!parsed.has_value()) {
            continue;
        }
        const std::string key = layer_key(*parsed, layer.node_id);
        layer_blob_agg & agg = blob_layers[key];
        agg.layer_index = *parsed;
        agg.node_id     = layer.node_id;
        agg.total_count++;
        agg.size_bytes += layer.size_bytes;
        if (!layer.device.empty()) {
            agg.device = layer.device;
        }
        if (layer.state == install_state::corrupted) {
            agg.any_corrupted = true;
        } else if (layer.state == install_state::ready) {
            agg.ready_count++;
        }
    }

    for (const auto & [key, agg] : blob_layers) {
        if (indexed.count(key)) {
            continue;
        }
        installed_layer layer;
        layer.layer_index = agg.layer_index;
        layer.node_id     = agg.node_id;
        layer.device      = agg.device;
        layer.size_bytes  = agg.size_bytes;
        if (agg.any_corrupted) {
            layer.state = install_state::corrupted;
        } else if (agg.ready_count > 0 && agg.ready_count == agg.total_count) {
            layer.state = install_state::ready;
        } else {
            layer.state = install_state::missing;
        }
        indexed[key] = layer;
    }
    return indexed;
}

static bool node_is_online(
        const std::string & node_id,
        const std::set<std::string> & online_nodes) {
    return online_nodes.empty() || online_nodes.count(node_id) > 0;
}

} // namespace

// ---------------------------------------------------------------------------
// install_state
// ---------------------------------------------------------------------------

std::string install_state_to_string(install_state s) {
    switch (s) {
        case install_state::missing:     return "MISSING";
        case install_state::downloading: return "DOWNLOADING";
        case install_state::verifying:   return "VERIFYING";
        case install_state::ready:       return "READY";
        case install_state::corrupted:   return "CORRUPTED";
    }
    return "MISSING";
}

install_state install_state_from_string(const std::string & s) {
    const std::string upper = upper_ascii(s);
    if (upper == "MISSING")     return install_state::missing;
    if (upper == "DOWNLOADING") return install_state::downloading;
    if (upper == "VERIFYING")   return install_state::verifying;
    if (upper == "READY")       return install_state::ready;
    if (upper == "CORRUPTED")   return install_state::corrupted;
    return install_state::missing;
}

// ---------------------------------------------------------------------------
// coverage_state
// ---------------------------------------------------------------------------

std::string coverage_state_to_string(coverage_state s) {
    switch (s) {
        case coverage_state::empty:    return "EMPTY";
        case coverage_state::partial:  return "PARTIAL";
        case coverage_state::ready:    return "READY";
        case coverage_state::degraded: return "DEGRADED";
    }
    return "EMPTY";
}

coverage_state coverage_state_from_string(const std::string & s) {
    const std::string upper = upper_ascii(s);
    if (upper == "EMPTY")    return coverage_state::empty;
    if (upper == "PARTIAL")  return coverage_state::partial;
    if (upper == "READY")    return coverage_state::ready;
    if (upper == "DEGRADED") return coverage_state::degraded;
    return coverage_state::empty;
}

// ---------------------------------------------------------------------------
// JSON
// ---------------------------------------------------------------------------

json installed_layer::to_json() const {
    json j = {
        { "layer", layer_index },
        { "layer_index", layer_index },
        { "node", node_id },
        { "node_id", node_id },
        { "device", device },
        { "size_bytes", size_bytes },
        { "checksum", checksum },
        { "state", install_state_to_string(state) },
    };
    if (!blob_id.empty()) {
        j["blob_id"] = blob_id;
    }
    if (!tensor_name.empty()) {
        j["tensor_name"] = tensor_name;
    }
    return j;
}

installed_layer installed_layer::from_json(const json & j) {
    installed_layer layer;
    layer.layer_index = j.value("layer", j.value("layer_index", -1));
    layer.node_id     = j.value("node", j.value("node_id", ""));
    layer.device      = j.value("device", "");
    layer.size_bytes  = j.value("size_bytes", static_cast<uint64_t>(0));
    layer.checksum    = j.value("checksum", "");
    layer.state       = install_state_from_string(j.value("state", "MISSING"));
    layer.blob_id     = j.value("blob_id", "");
    layer.tensor_name = j.value("tensor_name", "");
    layer.updated_at  = std::chrono::system_clock::now();
    return layer;
}

json actual_model_layout::to_json() const {
    json layers_json = json::array();
    for (const auto & layer : layers) {
        layers_json.push_back(layer.to_json());
    }
    return {
        { "model", model_id },
        { "model_id", model_id },
        { "layers", layers_json },
    };
}

actual_model_layout actual_model_layout::from_json(const json & j) {
    actual_model_layout layout;
    layout.model_id = j.value("model_id", j.value("model", ""));
    if (j.contains("layers") && j["layers"].is_array()) {
        for (const auto & item : j["layers"]) {
            layout.layers.push_back(installed_layer::from_json(item));
        }
    }
    return layout;
}

json coverage_report::to_json() const {
    json missing_json = json::array();
    for (int32_t layer : missing) {
        missing_json.push_back(layer);
    }
    json corrupted_json = json::array();
    for (int32_t layer : corrupted) {
        corrupted_json.push_back(layer);
    }
    return {
        { "model", model_id },
        { "model_id", model_id },
        { "state", coverage_state_to_string(state) },
        { "total_layers", total_layers },
        { "ready_layers", ready_layers },
        { "missing_layers", missing_layers },
        { "corrupted_layers", corrupted_layers },
        { "missing", missing_json },
        { "corrupted", corrupted_json },
    };
}

coverage_report coverage_report::from_json(const json & j) {
    coverage_report report;
    report.model_id = j.value("model_id", j.value("model", ""));
    report.state = coverage_state_from_string(j.value("state", "EMPTY"));
    report.total_layers = j.value("total_layers", 0);
    report.ready_layers = j.value("ready_layers", 0);
    report.missing_layers = j.value("missing_layers", 0);
    report.corrupted_layers = j.value("corrupted_layers", 0);
    if (j.contains("missing") && j["missing"].is_array()) {
        for (const auto & item : j["missing"]) {
            report.missing.push_back(item.get<int32_t>());
        }
    }
    if (j.contains("corrupted") && j["corrupted"].is_array()) {
        for (const auto & item : j["corrupted"]) {
            report.corrupted.push_back(item.get<int32_t>());
        }
    }
    return report;
}

json reconciliation_result::to_json() const {
    json missing_json = json::array();
    for (int32_t layer : missing) {
        missing_json.push_back(layer);
    }
    json corrupted_json = json::array();
    for (int32_t layer : corrupted) {
        corrupted_json.push_back(layer);
    }
    return {
        { "model", model_id },
        { "model_id", model_id },
        { "state", coverage_state_to_string(state) },
        { "missing", missing_json },
        { "corrupted", corrupted_json },
    };
}

reconciliation_result reconciliation_result::from_json(const json & j) {
    reconciliation_result result;
    result.model_id = j.value("model_id", j.value("model", ""));
    result.state = coverage_state_from_string(j.value("state", "EMPTY"));
    if (j.contains("missing") && j["missing"].is_array()) {
        for (const auto & item : j["missing"]) {
            result.missing.push_back(item.get<int32_t>());
        }
    }
    if (j.contains("corrupted") && j["corrupted"].is_array()) {
        for (const auto & item : j["corrupted"]) {
            result.corrupted.push_back(item.get<int32_t>());
        }
    }
    return result;
}

// ---------------------------------------------------------------------------
// Core algorithms
// ---------------------------------------------------------------------------

coverage_report compute_coverage(
        const desired_model_layout & desired,
        const actual_model_layout & actual,
        const std::set<std::string> & online_nodes) {
    coverage_report report;
    report.model_id = desired.model_id.empty() ? actual.model_id : desired.model_id;
    report.total_layers = static_cast<int>(desired.placements.size());

    const auto indexed = index_actual_layers(actual);
    bool node_loss_missing = false;

    for (const auto & placement : desired.placements) {
        const std::string key = layer_key(placement.layer_index, placement.node_id);
        const auto it = indexed.find(key);

        if (!node_is_online(placement.node_id, online_nodes)) {
            report.missing.push_back(placement.layer_index);
            node_loss_missing = true;
            continue;
        }

        if (it == indexed.end()) {
            report.missing.push_back(placement.layer_index);
            continue;
        }

        const installed_layer & layer = it->second;
        if (layer.state == install_state::corrupted) {
            report.corrupted.push_back(placement.layer_index);
            continue;
        }
        if (layer.state != install_state::ready) {
            report.missing.push_back(placement.layer_index);
            continue;
        }
        const std::string expected_device = dist_normalize_device(placement.device, placement.device != "cpu");
        const std::string actual_device   = dist_normalize_device(layer.device, layer.device != "cpu");
        if (!placement.device.empty() && !layer.device.empty() &&
                expected_device != actual_device) {
            report.missing.push_back(placement.layer_index);
            continue;
        }
        ++report.ready_layers;
    }

    std::sort(report.missing.begin(), report.missing.end());
    report.missing.erase(std::unique(report.missing.begin(), report.missing.end()),
            report.missing.end());
    std::sort(report.corrupted.begin(), report.corrupted.end());
    report.corrupted.erase(std::unique(report.corrupted.begin(), report.corrupted.end()),
            report.corrupted.end());

    report.missing_layers = static_cast<int>(report.missing.size());
    report.corrupted_layers = static_cast<int>(report.corrupted.size());

    if (report.total_layers == 0) {
        report.state = coverage_state::empty;
    } else if (report.corrupted_layers > 0) {
        report.state = coverage_state::degraded;
    } else if (node_loss_missing) {
        report.state = coverage_state::degraded;
    } else if (report.ready_layers == 0) {
        report.state = coverage_state::empty;
    } else if (report.missing_layers == 0 && report.ready_layers == report.total_layers) {
        report.state = coverage_state::ready;
    } else if (report.missing_layers > 0) {
        report.state = coverage_state::partial;
    } else {
        report.state = coverage_state::partial;
    }

    return report;
}

reconciliation_result reconcile_layers(
        const desired_model_layout & desired,
        const actual_model_layout & actual,
        const std::set<std::string> & online_nodes) {
    const coverage_report report = compute_coverage(desired, actual, online_nodes);
    reconciliation_result result;
    result.model_id = report.model_id;
    result.state = report.state;
    result.missing = report.missing;
    result.corrupted = report.corrupted;
    return result;
}

actual_model_layout merge_node_layer_reports(
        const std::string & model_id,
        const std::vector<actual_model_layout> & node_reports) {
    actual_model_layout merged;
    merged.model_id = model_id;
    std::set<std::string> seen;
    for (const auto & report : node_reports) {
        for (const auto & layer : report.layers) {
            const std::string key = actual_layer_key(layer);
            if (!seen.insert(key).second) {
                continue;
            }
            merged.layers.push_back(layer);
        }
    }
    std::sort(merged.layers.begin(), merged.layers.end(),
            [](const installed_layer & a, const installed_layer & b) {
                if (a.layer_index != b.layer_index) {
                    return a.layer_index < b.layer_index;
                }
                return a.node_id < b.node_id;
            });
    return merged;
}

actual_model_layout actual_layout_from_node_response(
        const std::string & node_id,
        const json & body) {
    actual_model_layout layout = actual_model_layout::from_json(body);
    if (layout.model_id.empty()) {
        layout.model_id = body.value("model", "");
    }
    for (auto & layer : layout.layers) {
        if (layer.node_id.empty()) {
            layer.node_id = node_id;
        }
        layer.updated_at = std::chrono::system_clock::now();
    }
    return layout;
}

bool coverage_has_no_extra_layers(
        const desired_model_layout & desired,
        const actual_model_layout & actual,
        std::string & error) {
    error.clear();
    std::set<std::tuple<int32_t, std::string>> desired_keys;
    for (const auto & p : desired.placements) {
        desired_keys.insert({ p.layer_index, p.node_id });
    }
    for (const auto & layer : actual.layers) {
        const auto key = std::make_tuple(layer.layer_index, layer.node_id);
        if (!desired_keys.count(key)) {
            error = "unexpected layer " + std::to_string(layer.layer_index) +
                    " on node " + layer.node_id;
            return false;
        }
    }
    return true;
}

std::optional<desired_model_layout> desired_layout_from_actual(
        const std::string & model_id,
        const actual_model_layout & actual,
        const int32_t n_layer) {
    if (n_layer <= 0) {
        return std::nullopt;
    }

    std::map<int32_t, installed_layer> ready_by_layer;
    for (const auto & layer : actual.layers) {
        if (layer.layer_index < 0 || layer.layer_index >= n_layer) {
            continue;
        }
        if (!layer.blob_id.empty() || !layer.tensor_name.empty()) {
            continue;
        }
        if (layer.state != install_state::ready) {
            continue;
        }
        if (layer.node_id.empty()) {
            continue;
        }
        ready_by_layer[layer.layer_index] = layer;
    }

    if (static_cast<int32_t>(ready_by_layer.size()) != n_layer) {
        return std::nullopt;
    }

    desired_model_layout layout;
    layout.model_id     = model_id;
    layout.fits_cluster = true;
    layout.placements.reserve(static_cast<size_t>(n_layer));

    for (int32_t i = 0; i < n_layer; ++i) {
        const auto it = ready_by_layer.find(i);
        if (it == ready_by_layer.end()) {
            return std::nullopt;
        }
        layer_placement p;
        p.layer_index = i;
        p.node_id     = it->second.node_id;
        p.device      = it->second.device;
        p.size_bytes  = it->second.size_bytes;
        p.required    = true;
        layout.total_weight_bytes += p.size_bytes;
        layout.placements.push_back(std::move(p));
    }

    std::sort(layout.placements.begin(), layout.placements.end(),
            [](const layer_placement & a, const layer_placement & b) {
                return a.layer_index < b.layer_index;
            });
    return layout;
}
