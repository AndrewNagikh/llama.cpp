#include "install_planning.h"

#include "semantic_blob.h"

#include <set>

namespace {

static std::string blob_download_key(
        const std::string & node_id,
        const std::string & blob_id,
        const std::string & tensor_name) {
    return blob_install_key(node_id, blob_id, tensor_name);
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

} // namespace

void add_semantic_blob_downloads(
        std::vector<install_operation> & operations,
        std::set<std::string> & seen,
        const architecture_descriptor & desc,
        const model_manifest & manifest,
        const std::string & entry_node,
        const std::string & final_node,
        const std::vector<std::string> & all_nodes,
        const std::string & source_url,
        const actual_model_layout & actual,
        const std::set<std::string> & ready_blobs) {
    (void) manifest;

    for (const semantic_blob & blob : desc.blobs) {
        if (blob.storage_alias || blob.deploy == blob_deploy_target::none || blob.tensors.empty()) {
            continue;
        }

        const std::vector<std::string> desired_nodes = nodes_for_blob_deploy(
                blob.deploy, entry_node, final_node, all_nodes);
        const std::string storage_id = blob.storage_alias && !blob.storage_blob_id.empty()
                ? blob.storage_blob_id
                : blob.id;

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
                if (ready_blobs.count(blob_download_key(node_id, storage_id, slot.name))) {
                    continue;
                }
                add_blob_download(operations, seen, node_id, storage_id, slot, source_url);
            }
        }
    }
}
