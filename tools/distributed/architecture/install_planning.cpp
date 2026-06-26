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
        const std::set<std::string> & ready_blobs) {
    (void) manifest;

    for (const semantic_blob & blob : desc.blobs) {
        if (blob.storage_alias || blob.deploy == blob_deploy_target::none || blob.tensors.empty()) {
            continue;
        }

        const std::vector<std::string> nodes = nodes_for_blob_deploy(
                blob.deploy, entry_node, final_node, all_nodes);
        const std::string storage_id = blob.storage_alias && !blob.storage_blob_id.empty()
                ? blob.storage_blob_id
                : blob.id;

        for (const std::string & node_id : nodes) {
            for (const semantic_tensor_slot & slot : blob.tensors) {
                if (ready_blobs.count(blob_download_key(node_id, storage_id, slot.name))) {
                    continue;
                }
                add_blob_download(operations, seen, node_id, storage_id, slot, source_url);
            }
        }
    }
}
