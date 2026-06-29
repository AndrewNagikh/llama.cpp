#include "store_verifier.h"

#include "architecture/semantic_runtime_descriptor.h"

namespace {

store_blob_entry make_entry(
        const std::string & blob_id,
        const std::string & tensor_name,
        const std::string & checksum,
        uint64_t size_bytes,
        const std::string & node_id,
        install_state state) {
    store_blob_entry entry;
    entry.blob_id     = blob_id;
    entry.tensor_name = tensor_name;
    entry.checksum    = checksum;
    entry.size_bytes  = size_bytes;
    entry.node_id     = node_id;
    entry.state       = state;
    return entry;
}

} // namespace

nlohmann::json store_verify_result::to_json() const {
    nlohmann::json blob_json = nlohmann::json::array();
    for (const auto & blob : blobs) {
        blob_json.push_back({
            { "blob_id", blob.blob_id },
            { "tensor_name", blob.tensor_name },
            { "checksum", blob.checksum },
            { "size_bytes", blob.size_bytes },
            { "node_id", blob.node_id },
            { "state", install_state_to_string(blob.state) },
        });
    }
    return {
        { "ok", ok },
        { "blob_count", blob_count },
        { "verified_count", verified_count },
        { "missing_count", missing_count },
        { "corrupted_count", corrupted_count },
        { "issues", issues },
        { "blobs", blob_json },
    };
}

nlohmann::json summarize_layer_store(const layer_store & store, const std::string & node_id) {
    store_verify_result summary;
    summary.node_id = node_id;

    for (const auto & layer : store.list_layers()) {
        install_state state = install_state::ready;
        if (!store.verify_layer(layer.layer_index, layer.checksum)) {
            state = install_state::corrupted;
            summary.corrupted_count++;
        } else {
            summary.verified_count++;
        }
        summary.blobs.push_back(make_entry(
                "layer:" + std::to_string(layer.layer_index),
                "",
                layer.checksum,
                layer.size_bytes,
                node_id,
                state));
    }

    for (const auto & blob : store.list_blob_tensors()) {
        const std::string checksum = blob.checksum.empty()
                ? ("manifest:tensor:" + blob.tensor_name)
                : blob.checksum;
        install_state state = install_state::ready;
        if (!store.verify_blob_tensor(blob.blob_id, blob.tensor_name, checksum)) {
            state = install_state::corrupted;
            summary.corrupted_count++;
        } else {
            summary.verified_count++;
        }
        summary.blobs.push_back(make_entry(
                blob.blob_id,
                blob.tensor_name,
                checksum,
                blob.size_bytes,
                node_id,
                state));
    }

    summary.blob_count = static_cast<int>(summary.blobs.size());
    summary.ok         = summary.corrupted_count == 0;
    return summary.to_json();
}

store_verify_result verify_layer_store(
        const layer_store & store,
        const model_manifest & manifest,
        const std::string & node_id) {
    store_verify_result result;
    if (manifest.empty()) {
        result.issues.push_back("empty manifest");
        return result;
    }

    const semantic_runtime_descriptor rt = build_semantic_runtime_descriptor(manifest);

    for (const auto & blob : rt.blobs) {
        if (blob.storage_alias || blob.tensors.empty()) {
            continue;
        }
        const std::string storage_id = blob.storage_blob_id.empty() ? blob.id : blob.storage_blob_id;
        for (const auto & slot : blob.tensors) {
            result.blob_count++;
            const std::string checksum = "manifest:tensor:" + slot.name;
            if (!store.has_blob_tensor(storage_id, slot.name)) {
                result.missing_count++;
                result.issues.push_back("missing blob tensor " + storage_id + "/" + slot.name);
                result.blobs.push_back(make_entry(
                        storage_id, slot.name, checksum, slot.size_bytes, node_id,
                        install_state::missing));
                continue;
            }
            if (!store.verify_blob_tensor(storage_id, slot.name, checksum)) {
                result.corrupted_count++;
                result.issues.push_back("checksum failed " + storage_id + "/" + slot.name);
                result.blobs.push_back(make_entry(
                        storage_id, slot.name, checksum, slot.size_bytes, node_id,
                        install_state::corrupted));
                continue;
            }
            result.verified_count++;
            result.blobs.push_back(make_entry(
                    storage_id, slot.name, checksum, slot.size_bytes, node_id,
                    install_state::ready));
        }
    }

    result.ok = result.missing_count == 0 && result.corrupted_count == 0;
    return result;
}

store_verify_result compare_store_to_actual(
        const store_verify_result & store_view,
        const actual_model_layout & actual,
        const std::string & node_id) {
    store_verify_result result = store_view;

    auto find_actual = [&](const std::string & blob_id, const std::string & tensor_name) {
        for (const auto & layer : actual.layers) {
            if (layer.node_id != node_id) {
                continue;
            }
            if (layer.blob_id == blob_id &&
                    (tensor_name.empty() || layer.tensor_name == tensor_name)) {
                return layer.state;
            }
        }
        return install_state::missing;
    };

    for (const auto & blob : store_view.blobs) {
        const install_state actual_state = find_actual(blob.blob_id, blob.tensor_name);
        if (actual_state == install_state::missing && blob.state == install_state::ready) {
            result.issues.push_back(
                    "registry missing store blob " + blob.blob_id + "/" + blob.tensor_name);
            result.ok = false;
        }
        if (actual_state == install_state::corrupted && blob.state == install_state::ready) {
            result.issues.push_back(
                    "registry corrupted but store ready " + blob.blob_id + "/" + blob.tensor_name);
            result.ok = false;
        }
    }

    return result;
}
