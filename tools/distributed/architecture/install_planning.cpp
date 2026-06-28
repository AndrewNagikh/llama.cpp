#include "install_planning.h"

#include "coverage/coverage.h"
#include "semantic_blob.h"

#include <algorithm>
#include <map>
#include <set>

namespace {

static std::string blob_download_key(
        const std::string & node_id,
        const std::string & blob_id,
        const std::string & tensor_name) {
    return blob_install_key(node_id, blob_id, tensor_name);
}

static std::string layer_node_key(int32_t layer_index, const std::string & node_id) {
    return std::to_string(layer_index) + "@" + node_id;
}

static bool contains_layer(const std::vector<int32_t> & layers, int32_t layer_index) {
    return std::find(layers.begin(), layers.end(), layer_index) != layers.end();
}

static download_operation make_tensor_download(
        const std::string & node_id,
        const std::string & blob_id,
        const semantic_tensor_slot & slot,
        const std::string & source_url) {
    download_operation op;
    op.node_id       = node_id;
    op.blob_id       = blob_id;
    op.tensor_name   = slot.name;
    op.tensor_offset = slot.offset;
    op.tensor_length = slot.size_bytes;
    op.source_url    = source_url;
    op.checksum      = "manifest:tensor:" + slot.name;
    return op;
}

static void add_blob_download(
        std::vector<install_operation> & operations,
        std::set<std::string> & seen,
        const std::string & node_id,
        const std::string & blob_id,
        const semantic_tensor_slot & slot,
        const std::string & source_url) {
    const std::string key = blob_download_key(node_id, blob_id, slot.name);
    if (!seen.insert(key).second) {
        return;
    }

    install_operation op;
    op.action      = install_action::download;
    op.node_id     = node_id;
    op.layer_index = -1;
    op.download    = make_tensor_download(node_id, blob_id, slot, source_url);
    operations.push_back(std::move(op));
}

static void add_blob_repair(
        std::vector<install_operation> & operations,
        std::set<std::string> & seen,
        const std::string & node_id,
        const std::string & blob_id,
        const semantic_tensor_slot & slot,
        const std::string & source_url) {
    const std::string key = "repair:" + blob_download_key(node_id, blob_id, slot.name);
    if (!seen.insert(key).second) {
        return;
    }

    install_operation op;
    op.action      = install_action::repair;
    op.node_id     = node_id;
    op.layer_index = -1;
    op.download    = make_tensor_download(node_id, blob_id, slot, source_url);
    operations.push_back(std::move(op));
}

static void add_blob_delete(
        std::vector<install_operation> & operations,
        std::set<std::string> & seen,
        const std::string & node_id,
        const std::string & blob_id,
        const semantic_tensor_slot & slot) {
    const std::string key = "delete:" + blob_download_key(node_id, blob_id, slot.name);
    if (!seen.insert(key).second) {
        return;
    }

    install_operation op;
    op.action               = install_action::delete_op;
    op.node_id              = node_id;
    op.layer_index          = -1;
    op.download.blob_id     = blob_id;
    op.download.tensor_name = slot.name;
    op.download.node_id     = node_id;
    operations.push_back(std::move(op));
}

static void reconcile_blob(
        std::vector<install_operation> & operations,
        std::set<std::string> & seen,
        const semantic_blob & blob,
        const std::string & storage_id,
        const std::vector<std::string> & desired_nodes,
        const std::string & source_url,
        const actual_model_layout & actual,
        const std::set<std::string> & ready_blobs) {
    std::set<std::string> desired_set(desired_nodes.begin(), desired_nodes.end());

    for (const auto & layer : actual.layers) {
        if (layer.blob_id != storage_id && layer.blob_id != blob.id) {
            continue;
        }
        if (layer.tensor_name.empty()) {
            continue;
        }
        if (layer.state != install_state::ready && layer.state != install_state::corrupted) {
            continue;
        }
        if (desired_set.count(layer.node_id)) {
            continue;
        }
        for (const semantic_tensor_slot & slot : blob.tensors) {
            if (slot.name != layer.tensor_name) {
                continue;
            }
            add_blob_delete(operations, seen, layer.node_id, storage_id, slot);
        }
    }

    for (const std::string & node_id : desired_nodes) {
        for (const semantic_tensor_slot & slot : blob.tensors) {
            const std::string key = blob_download_key(node_id, storage_id, slot.name);
            if (ready_blobs.count(key)) {
                continue;
            }
            bool ready_on_node = false;
            bool corrupted       = false;
            for (const auto & layer : actual.layers) {
                if (layer.node_id != node_id) {
                    continue;
                }
                if (layer.blob_id != storage_id && layer.blob_id != blob.id) {
                    continue;
                }
                if (layer.tensor_name != slot.name) {
                    continue;
                }
                if (layer.state == install_state::ready) {
                    ready_on_node = true;
                }
                if (layer.state == install_state::corrupted) {
                    corrupted = true;
                }
            }
            if (ready_on_node) {
                continue;
            }
            if (corrupted) {
                add_blob_repair(operations, seen, node_id, storage_id, slot, source_url);
            } else {
                add_blob_download(operations, seen, node_id, storage_id, slot, source_url);
            }
        }
    }
}

} // namespace

void add_semantic_blob_downloads(
        std::vector<install_operation> & operations,
        std::set<std::string> & seen,
        const semantic_runtime_descriptor & rt,
        const std::string & entry_node,
        const std::string & final_node,
        const std::vector<std::string> & all_nodes,
        const std::string & source_url,
        const actual_model_layout & actual,
        const std::set<std::string> & ready_blobs) {
    for (const semantic_blob & blob : rt.blobs) {
        if (blob.storage_alias || blob.deploy == blob_deploy_target::none || blob.tensors.empty()) {
            continue;
        }
        if (blob.role == tensor_semantic_role::transformer_layer) {
            continue;
        }

        const std::vector<std::string> desired_nodes = nodes_for_blob_deploy(
                blob.deploy, entry_node, final_node, all_nodes);
        const std::string storage_id = blob.storage_alias && !blob.storage_blob_id.empty()
                ? blob.storage_blob_id
                : blob.id;

        reconcile_blob(
                operations,
                seen,
                blob,
                storage_id,
                desired_nodes,
                source_url,
                actual,
                ready_blobs);
    }
}

void add_layer_blob_downloads(
        std::vector<install_operation> & operations,
        std::set<std::string> & seen,
        const semantic_runtime_descriptor & rt,
        const desired_model_layout & desired,
        const actual_model_layout & actual,
        const coverage_report & coverage,
        const std::string & source_url) {
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

    for (const auto & placement : desired.placements) {
        const std::string blob_id = layer_blob_id(placement.layer_index);
        const semantic_blob * blob = find_blob(rt.blobs, blob_id);
        if (blob == nullptr || blob->tensors.empty()) {
            continue;
        }

        for (const auto & layer : actual.layers) {
            if (layer.blob_id != blob_id || layer.tensor_name.empty()) {
                continue;
            }
            if (layer.node_id == placement.node_id) {
                continue;
            }
            if (layer.state == install_state::ready ||
                    layer.state == install_state::corrupted) {
                for (const semantic_tensor_slot & slot : blob->tensors) {
                    if (slot.name == layer.tensor_name) {
                        add_blob_delete(operations, seen, layer.node_id, blob_id, slot);
                    }
                }
            }
        }

        if (!contains_layer(coverage.missing, placement.layer_index) &&
                !contains_layer(coverage.corrupted, placement.layer_index)) {
            continue;
        }

        const bool repair = contains_layer(coverage.corrupted, placement.layer_index);

        for (const semantic_tensor_slot & slot : blob->tensors) {
            const std::string key = blob_download_key(placement.node_id, blob_id, slot.name);
            if (ready_blobs.count(key) && !repair) {
                continue;
            }
            if (repair) {
                add_blob_repair(
                        operations, seen, placement.node_id, blob_id, slot, source_url);
            } else {
                add_blob_download(
                        operations, seen, placement.node_id, blob_id, slot, source_url);
            }
        }
    }
}
