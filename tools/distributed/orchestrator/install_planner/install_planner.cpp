#include "install_planner.h"

#include "architecture/install_planning.h"
#include "architecture/semantic_runtime_descriptor.h"
#include "coverage/runtime_coverage.h"

#include <algorithm>
#include <cctype>
#include <climits>
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

static std::string layer_node_key(int32_t layer_index, const std::string & node_id) {
    return std::to_string(layer_index) + "@" + node_id;
}

static bool contains_layer(const std::vector<int32_t> & layers, int32_t layer_index) {
    return std::find(layers.begin(), layers.end(), layer_index) != layers.end();
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

static std::optional<int32_t> operation_layer_index(const install_operation & op) {
    if (op.layer_index >= 0) {
        return op.layer_index;
    }
    return layer_index_from_blob_id(op.download.blob_id);
}

static download_operation make_download_op(
        int32_t layer_index,
        const std::string & node_id,
        const layer_byte_range & range,
        const std::string & source_url) {
    download_operation op;
    op.layer_index   = layer_index;
    op.node_id       = node_id;
    op.tensor_offset = range.offset;
    op.tensor_length = range.length > 0 ? range.length : range.size_bytes;
    op.checksum      = range.checksum;
    op.source_url    = source_url;
    return op;
}

static void add_operation(
        std::vector<install_operation> & operations,
        std::set<std::tuple<int, std::string, int32_t>> & seen,
        install_action action,
        const std::string & node_id,
        int32_t layer_index,
        const download_operation & download = {}) {
    const int action_key = static_cast<int>(action);
    const auto key = std::make_tuple(action_key, node_id, layer_index);
    if (!seen.insert(key).second) {
        return;
    }
    install_operation op;
    op.action      = action;
    op.node_id     = node_id;
    op.layer_index = layer_index;
    op.download    = download;
    if (action == install_action::download || action == install_action::repair) {
        op.download.layer_index = layer_index;
        op.download.node_id     = node_id;
    }
    operations.push_back(std::move(op));
}

static void finalize_plan(install_plan & plan) {
    std::sort(plan.operations.begin(), plan.operations.end(),
            [](const install_operation & a, const install_operation & b) {
                if (a.node_id != b.node_id) {
                    return a.node_id < b.node_id;
                }
                if (a.action != b.action) {
                    return static_cast<int>(a.action) < static_cast<int>(b.action);
                }
                return a.layer_index < b.layer_index;
            });

    plan.groups = group_install_operations(plan.operations);
    plan.operation_count = static_cast<int>(plan.operations.size());
    plan.total_download_bytes = 0;
    for (const auto & op : plan.operations) {
        if (op.action == install_action::download || op.action == install_action::repair) {
            plan.total_download_bytes += op.download.tensor_length;
        }
    }
}

} // namespace

// ---------------------------------------------------------------------------
// install_action
// ---------------------------------------------------------------------------

std::string install_action_to_string(install_action action) {
    switch (action) {
        case install_action::download: return "DOWNLOAD";
        case install_action::verify:   return "VERIFY";
        case install_action::delete_op: return "DELETE";
        case install_action::repair:   return "REPAIR";
    }
    return "DOWNLOAD";
}

install_action install_action_from_string(const std::string & s) {
    const std::string upper = upper_ascii(s);
    if (upper == "DOWNLOAD") return install_action::download;
    if (upper == "VERIFY")   return install_action::verify;
    if (upper == "DELETE")   return install_action::delete_op;
    if (upper == "REPAIR")   return install_action::repair;
    return install_action::download;
}

// ---------------------------------------------------------------------------
// JSON
// ---------------------------------------------------------------------------

json download_operation::to_json() const {
    json j = {
        { "layer", layer_index },
        { "node", node_id },
        { "tensor_offset", tensor_offset },
        { "tensor_length", tensor_length },
        { "offset", tensor_offset },
        { "length", tensor_length },
        { "source_url", source_url },
        { "checksum", checksum },
    };
    if (!blob_id.empty()) {
        j["blob_id"] = blob_id;
    }
    if (!tensor_name.empty()) {
        j["tensor_name"] = tensor_name;
    }
    return j;
}

download_operation download_operation::from_json(const json & j) {
    download_operation op;
    op.layer_index   = j.value("layer", j.value("layer_index", -1));
    op.blob_id       = j.value("blob_id", "");
    op.tensor_name   = j.value("tensor_name", "");
    op.node_id       = j.value("node", j.value("node_id", ""));
    op.tensor_offset = j.value("tensor_offset", j.value("offset", static_cast<uint64_t>(0)));
    op.tensor_length = j.value("tensor_length", j.value("length", static_cast<uint64_t>(0)));
    op.source_url    = j.value("source_url", "");
    op.checksum      = j.value("checksum", "");
    return op;
}

json install_operation::to_json() const {
    json j = {
        { "action", install_action_to_string(action) },
        { "node", node_id },
        { "layer", layer_index },
    };
    if (action == install_action::download || action == install_action::repair) {
        j["offset"]        = download.tensor_offset;
        j["length"]        = download.tensor_length;
        j["tensor_offset"] = download.tensor_offset;
        j["tensor_length"] = download.tensor_length;
        j["source_url"]    = download.source_url;
        j["checksum"]      = download.checksum;
        j["download"]      = download.to_json();
    } else if (action == install_action::delete_op &&
            (!download.blob_id.empty() || !download.tensor_name.empty())) {
        j["download"] = download.to_json();
    }
    return j;
}

install_operation install_operation::from_json(const json & j) {
    install_operation op;
    op.action      = install_action_from_string(j.value("action", "DOWNLOAD"));
    op.node_id     = j.value("node", j.value("node_id", ""));
    op.layer_index = j.value("layer", j.value("layer_index", -1));
    if (j.contains("download") && j["download"].is_object()) {
        op.download = download_operation::from_json(j["download"]);
    } else {
        op.download = download_operation::from_json(j);
    }
    op.download.layer_index = op.layer_index;
    op.download.node_id     = op.node_id;
    return op;
}

json install_plan_group::to_json() const {
    json layers_json = json::array();
    for (int32_t layer : layers) {
        layers_json.push_back(layer);
    }
    return {
        { "node", node_id },
        { "action", install_action_to_string(action) },
        { "layers", layers_json },
        { "download_bytes", download_bytes },
    };
}

install_plan_group install_plan_group::from_json(const json & j) {
    install_plan_group group;
    group.node_id = j.value("node", j.value("node_id", ""));
    group.action  = install_action_from_string(j.value("action", "DOWNLOAD"));
    group.download_bytes = j.value("download_bytes", static_cast<uint64_t>(0));
    if (j.contains("layers") && j["layers"].is_array()) {
        for (const auto & item : j["layers"]) {
            group.layers.push_back(item.get<int32_t>());
        }
    }
    return group;
}

json install_plan::to_json() const {
    json operations_json = json::array();
    for (const auto & op : operations) {
        operations_json.push_back(op.to_json());
    }
    json groups_json = json::array();
    for (const auto & group : groups) {
        groups_json.push_back(group.to_json());
    }
    return {
        { "model", model_id },
        { "model_id", model_id },
        { "operations", operations_json },
        { "groups", groups_json },
        { "operation_count", operation_count },
        { "total_download_bytes", total_download_bytes },
    };
}

install_plan install_plan::from_json(const json & j) {
    install_plan plan;
    plan.model_id = j.value("model_id", j.value("model", ""));
    plan.operation_count = j.value("operation_count", 0);
    plan.total_download_bytes = j.value("total_download_bytes", static_cast<uint64_t>(0));
    if (j.contains("operations") && j["operations"].is_array()) {
        for (const auto & item : j["operations"]) {
            plan.operations.push_back(install_operation::from_json(item));
        }
    }
    if (j.contains("groups") && j["groups"].is_array()) {
        for (const auto & item : j["groups"]) {
            plan.groups.push_back(install_plan_group::from_json(item));
        }
    }
    if (plan.operation_count == 0) {
        plan.operation_count = static_cast<int>(plan.operations.size());
    }
    return plan;
}

// ---------------------------------------------------------------------------
// Manifest helpers
// ---------------------------------------------------------------------------

layer_byte_range manifest_layer_byte_range(
        const model_manifest & manifest,
        int32_t layer_index) {
    layer_byte_range range{};
    uint64_t min_offset = UINT64_MAX;
    uint64_t max_end    = 0;
    uint64_t sum_bytes  = 0;

    for (const auto & t : manifest.tensors) {
        if (t.layer != layer_index) {
            continue;
        }
        sum_bytes += t.size_bytes;
        if (t.offset > 0 || t.size_bytes > 0) {
            min_offset = std::min(min_offset, t.offset);
            max_end    = std::max(max_end, t.offset + t.size_bytes);
        }
    }

    if (min_offset != UINT64_MAX && max_end > min_offset) {
        range.offset     = min_offset;
        range.length     = max_end - min_offset;
        range.size_bytes = range.length;
    } else {
        for (const auto & ld : manifest.layers) {
            if (ld.layer_index == layer_index) {
                range.size_bytes = ld.size_bytes;
                range.length     = ld.size_bytes;
                break;
            }
        }
    }

    if (range.size_bytes == 0) {
        range.size_bytes = sum_bytes;
        range.length     = sum_bytes;
    }

    range.checksum = "manifest:layer:" + std::to_string(layer_index);
    return range;
}

layer_byte_range manifest_role_byte_range(
        const model_manifest & manifest,
        const tensor_role role) {
    layer_byte_range range{};
    uint64_t min_offset = UINT64_MAX;
    uint64_t max_end    = 0;

    for (const auto & t : manifest.tensors) {
        if (t.role != role) {
            continue;
        }
        if (t.offset > 0 || t.size_bytes > 0) {
            min_offset = std::min(min_offset, t.offset);
            max_end    = std::max(max_end, t.offset + t.size_bytes);
        }
    }

    if (min_offset == UINT64_MAX || max_end <= min_offset) {
        return range;
    }

    range.offset     = min_offset;
    range.length     = max_end - min_offset;
    range.size_bytes = range.length;
    range.checksum   = "manifest:role:" + tensor_role_to_string(role);
    return range;
}

layer_byte_range manifest_global_preamble_range(const model_manifest & manifest) {
    layer_byte_range range{};
    uint64_t min_offset = UINT64_MAX;
    uint64_t max_end    = 0;

    int32_t first_layer = INT32_MAX;
    for (const auto & ld : manifest.layers) {
        first_layer = std::min(first_layer, ld.layer_index);
    }
    if (first_layer == INT32_MAX && manifest.n_layer > 0) {
        first_layer = 0;
    }

    for (const auto & t : manifest.tensors) {
        if (t.layer >= first_layer) {
            continue;
        }
        if (t.role == tensor_role::output_norm || t.role == tensor_role::lm_head) {
            continue;
        }
        if (t.offset > 0 || t.size_bytes > 0) {
            min_offset = std::min(min_offset, t.offset);
            max_end    = std::max(max_end, t.offset + t.size_bytes);
        }
    }

    if (min_offset != UINT64_MAX && max_end > min_offset) {
        range.offset     = min_offset;
        range.length     = max_end - min_offset;
        range.size_bytes = range.length;
        range.checksum   = "manifest:role:preamble";
    }
    return range;
}

// ---------------------------------------------------------------------------
// Planner
// ---------------------------------------------------------------------------

install_plan_build_result build_install_plan(
        const model_manifest & manifest,
        const desired_model_layout & desired,
        const actual_model_layout & actual,
        const coverage_report & coverage,
        const std::string & source_url,
        const runtime_install_node_map * runtime_nodes) {
    install_plan_build_result result{};
    result.plan.model_id = desired.model_id.empty() ? coverage.model_id : desired.model_id;

    if (manifest.empty() || desired.placements.empty()) {
        result.error = "manifest or desired layout is empty";
        return result;
    }

    std::vector<install_operation> operations;
    std::set<std::string> blob_seen;

    const semantic_runtime_descriptor rt = build_semantic_runtime_descriptor(manifest);

    // Task 9.9 — idempotent fast path: fully ready layout needs zero operations.
    {
        const runtime_coverage_report rt_cov = compute_runtime_coverage(
                rt, desired, actual, {}, runtime_nodes);
        if (rt_cov.fully_ready()) {
            result.success = true;
            finalize_plan(result.plan);
            return result;
        }
    }

    std::string entry_node;
    std::string final_node;
    const int32_t last_layer = static_cast<int32_t>(manifest.n_layer) - 1;

    for (const auto & placement : desired.placements) {
        if (placement.layer_index == 0) {
            entry_node = placement.node_id;
        }
        if (placement.layer_index == last_layer) {
            final_node = placement.node_id;
        }
    }

    if (entry_node.empty() || final_node.empty()) {
        int32_t min_layer = INT32_MAX;
        int32_t max_layer = -1;
        for (const auto & placement : desired.placements) {
            if (placement.layer_index < min_layer) {
                min_layer  = placement.layer_index;
                entry_node = placement.node_id;
            }
            if (placement.layer_index > max_layer) {
                max_layer  = placement.layer_index;
                final_node = placement.node_id;
            }
        }
    }

    std::set<std::string> all_nodes;
    for (const auto & placement : desired.placements) {
        all_nodes.insert(placement.node_id);
    }
    std::vector<std::string> node_list(all_nodes.begin(), all_nodes.end());

    std::set<std::string> ready_blobs;
    for (const auto & layer : actual.layers) {
        if (layer.blob_id.empty() || layer.tensor_name.empty()) {
            continue;
        }
        if (layer.state != install_state::ready) {
            continue;
        }
        ready_blobs.insert(blob_install_key(
                layer.node_id, layer.blob_id, layer.tensor_name));
    }

    add_layer_blob_downloads(
            operations,
            blob_seen,
            rt,
            desired,
            actual,
            coverage,
            source_url);

    if (!entry_node.empty() || !final_node.empty()) {
        const runtime_install_node_map nodes = runtime_nodes != nullptr && runtime_nodes->valid()
                ? *runtime_nodes
                : runtime_install_node_map_legacy(entry_node, final_node, node_list);
        add_semantic_blob_downloads(
                operations,
                blob_seen,
                rt,
                nodes,
                source_url,
                actual,
                ready_blobs);
    }

    result.plan.operations = std::move(operations);
    finalize_plan(result.plan);
    result.success = true;
    return result;
}

std::vector<install_plan_group> group_install_operations(
        const std::vector<install_operation> & operations) {
    std::vector<install_plan_group> groups;
    if (operations.empty()) {
        return groups;
    }

    install_plan_group current;
    current.node_id = operations.front().node_id;
    current.action  = operations.front().action;

    auto flush = [&]() {
        if (!current.layers.empty()) {
            groups.push_back(current);
        }
        current.layers.clear();
        current.download_bytes = 0;
    };

    for (const auto & op : operations) {
        const bool same_group = op.node_id == current.node_id &&
                op.action == current.action &&
                !current.layers.empty() &&
                op.layer_index == current.layers.back() + 1;

        if (!current.layers.empty() && !same_group) {
            flush();
            current.node_id = op.node_id;
            current.action  = op.action;
        } else if (current.layers.empty()) {
            current.node_id = op.node_id;
            current.action  = op.action;
        }

        current.layers.push_back(op.layer_index);
        if (op.action == install_action::download || op.action == install_action::repair) {
            current.download_bytes += op.download.tensor_length;
        }
    }
    flush();
    return groups;
}

bool validate_install_plan(
        const install_plan & plan,
        const desired_model_layout & desired,
        const coverage_report & coverage,
        std::string & error) {
    error.clear();

    std::set<int32_t> ready_layers;
    for (const auto & placement : desired.placements) {
        if (!contains_layer(coverage.missing, placement.layer_index) &&
                !contains_layer(coverage.corrupted, placement.layer_index)) {
            ready_layers.insert(placement.layer_index);
        }
    }

    std::set<int32_t> planned_download;
    std::set<int32_t> planned_repair;
    for (const auto & op : plan.operations) {
        const std::optional<int32_t> op_layer = operation_layer_index(op);
        if (op.action == install_action::download) {
            if (op_layer.has_value() && ready_layers.count(*op_layer)) {
                error = "download planned for ready layer " + std::to_string(*op_layer);
                return false;
            }
            if (op.download.tensor_length == 0) {
                const std::string layer_label = op_layer.has_value()
                        ? std::to_string(*op_layer)
                        : op.download.blob_id;
                error = "download missing tensor length for layer " + layer_label;
                return false;
            }
            if (op_layer.has_value()) {
                planned_download.insert(*op_layer);
            }
        } else if (op.action == install_action::repair) {
            if (op_layer.has_value() &&
                    !contains_layer(coverage.corrupted, *op_layer)) {
                error = "repair planned for non-corrupted layer " + std::to_string(*op_layer);
                return false;
            }
            if (op_layer.has_value()) {
                planned_repair.insert(*op_layer);
            }
            if (op.download.tensor_length == 0) {
                const std::string layer_label = op_layer.has_value()
                        ? std::to_string(*op_layer)
                        : op.download.blob_id;
                error = "repair missing tensor length for layer " + layer_label;
                return false;
            }
        }
    }

    for (int32_t layer : coverage.missing) {
        if (planned_download.count(layer) == 0) {
            error = "missing layer " + std::to_string(layer) + " has no install operation";
            return false;
        }
    }

    for (int32_t layer : coverage.corrupted) {
        if (planned_repair.count(layer) == 0) {
            error = "corrupted layer " + std::to_string(layer) + " has no repair operation";
            return false;
        }
    }

    if (plan.operation_count != static_cast<int>(plan.operations.size())) {
        error = "operation_count mismatch";
        return false;
    }

    return true;
}

bool install_plan_has_grouped_layers(
        const install_plan & plan,
        const std::string & node_id,
        install_action action,
        const std::vector<int32_t> & expected_layers) {
    for (const auto & group : plan.groups) {
        if (group.node_id != node_id || group.action != action) {
            continue;
        }
        if (group.layers == expected_layers) {
            return true;
        }
    }
    return false;
}
