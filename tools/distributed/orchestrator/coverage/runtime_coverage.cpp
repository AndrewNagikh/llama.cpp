#include "runtime_coverage.h"

#include "architecture/semantic_runtime_descriptor.h"

#include <algorithm>
#include <climits>
#include <map>
#include <set>

using json = nlohmann::json;

namespace {

std::string layout_entry_final(
        const desired_model_layout & desired,
        bool entry_not_final) {
    if (desired.placements.empty()) {
        return {};
    }
    if (entry_not_final) {
        int32_t min_layer = INT32_MAX;
        std::string node;
        for (const auto & p : desired.placements) {
            if (p.layer_index < min_layer) {
                min_layer = p.layer_index;
                node      = p.node_id;
            }
        }
        return node;
    }
    int32_t max_layer = -1;
    std::string node;
    for (const auto & p : desired.placements) {
        if (p.layer_index > max_layer) {
            max_layer = p.layer_index;
            node      = p.node_id;
        }
    }
    return node;
}

std::pair<std::string, std::string> layout_entry_final_nodes(
        const desired_model_layout & desired) {
    std::string entry;
    std::string final_node;
    for (const auto & p : desired.placements) {
        if (p.layer_index == 0) {
            entry = p.node_id;
        }
    }
    if (desired.placements.size() > 0) {
        int32_t last = -1;
        for (const auto & p : desired.placements) {
            if (p.layer_index > last) {
                last       = p.layer_index;
                final_node = p.node_id;
            }
        }
    }
    if (entry.empty()) {
        entry = layout_entry_final(desired, true);
    }
    if (final_node.empty()) {
        final_node = layout_entry_final(desired, false);
    }
    return { entry, final_node };
}

bool blob_ready_on_node(
        const actual_model_layout & actual,
        const std::string & node_id,
        const std::string & blob_id,
        const std::string & tensor_name) {
    for (const auto & layer : actual.layers) {
        if (layer.node_id != node_id) {
            continue;
        }
        if (layer.blob_id != blob_id) {
            continue;
        }
        if (!tensor_name.empty() && layer.tensor_name != tensor_name) {
            continue;
        }
        if (layer.state == install_state::ready) {
            return true;
        }
    }
    return false;
}

} // namespace

json runtime_coverage_report::to_json() const {
    json j = layer_coverage.to_json();
    j["storage_state"] = coverage_state_to_string(storage_state);
    j["runtime_state"]  = coverage_state_to_string(runtime_state);
    j["missing_blobs"]  = missing_blobs;
    j["misplaced_blobs"] = misplaced_blobs;
    return j;
}

runtime_coverage_report compute_runtime_coverage(
        const semantic_runtime_descriptor & rt,
        const desired_model_layout & desired,
        const actual_model_layout & actual,
        const std::set<std::string> & online_nodes) {
    runtime_coverage_report report;
    report.layer_coverage = compute_coverage(desired, actual, online_nodes);

    if (rt.empty() || desired.placements.empty()) {
        report.storage_state = coverage_state::empty;
        report.runtime_state = coverage_state::empty;
        return report;
    }

    const auto [entry_node, final_node] = layout_entry_final_nodes(desired);

    std::set<std::string> layout_nodes;
    for (const auto & p : desired.placements) {
        layout_nodes.insert(p.node_id);
    }
    std::vector<std::string> node_list(layout_nodes.begin(), layout_nodes.end());

    int storage_missing = 0;
    int storage_ready   = 0;
    int storage_total   = 0;

    for (const semantic_blob & blob : rt.blobs) {
        if (blob.storage_alias || blob.deploy == blob_deploy_target::none ||
                blob.tensors.empty()) {
            continue;
        }

        const std::vector<std::string> targets = nodes_for_blob_deploy(
                blob.deploy, entry_node, final_node, node_list);
        const std::string storage_id = blob.storage_alias && !blob.storage_blob_id.empty()
                ? blob.storage_blob_id
                : blob.id;

        for (const std::string & node_id : targets) {
            for (const auto & slot : blob.tensors) {
                ++storage_total;
                if (blob_ready_on_node(actual, node_id, storage_id, slot.name)) {
                    ++storage_ready;
                    continue;
                }
                report.missing_blobs.push_back(node_id + ":" + storage_id + "/" + slot.name);
                ++storage_missing;
            }
        }

        for (const auto & layer : actual.layers) {
            if (layer.blob_id != storage_id && layer.blob_id != blob.id) {
                continue;
            }
            if (layer.tensor_name.empty()) {
                continue;
            }
            if (layer.state != install_state::ready) {
                continue;
            }
            if (std::find(targets.begin(), targets.end(), layer.node_id) != targets.end()) {
                continue;
            }
            report.misplaced_blobs.push_back(
                    layer.node_id + ":" + layer.blob_id + "/" + layer.tensor_name);
        }
    }

    if (storage_total == 0) {
        report.storage_state = coverage_state::empty;
    } else if (storage_missing == 0) {
        report.storage_state = coverage_state::ready;
    } else if (storage_ready == 0) {
        report.storage_state = coverage_state::empty;
    } else {
        report.storage_state = coverage_state::partial;
    }

    if (report.layer_coverage.state == coverage_state::ready &&
            report.storage_state == coverage_state::ready &&
            report.misplaced_blobs.empty()) {
        report.runtime_state = coverage_state::ready;
    } else if (report.layer_coverage.state == coverage_state::degraded ||
            !report.misplaced_blobs.empty()) {
        report.runtime_state = coverage_state::degraded;
    } else if (report.layer_coverage.state == coverage_state::ready &&
            report.storage_state == coverage_state::partial) {
        report.runtime_state = coverage_state::partial;
    } else if (report.layer_coverage.state == coverage_state::ready &&
            report.storage_state == coverage_state::ready &&
            !report.misplaced_blobs.empty()) {
        report.runtime_state = coverage_state::degraded;
    } else {
        report.runtime_state = coverage_state::partial;
    }

    if (report.fully_ready()) {
        report.layer_coverage.state = coverage_state::ready;
    } else if (report.runtime_state == coverage_state::degraded) {
        report.layer_coverage.state = coverage_state::degraded;
    }

    return report;
}
