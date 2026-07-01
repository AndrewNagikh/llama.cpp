#include "http_range_executor.h"

#include "dist_http_fetch.h"
#include "node_agent/layer_store/layer_checksum.h"

#include <vector>

namespace {

static bool fetch_range(
        const std::string & source_url,
        const uint64_t offset,
        const uint64_t length,
        std::vector<uint8_t> & out) {
    return dist_http_get_range(source_url, offset, length, out);
}

} // namespace

bool http_range_download_executor::fetch_body(
        const download_operation & dl,
        std::vector<uint8_t> & out,
        std::string & error) {
    if (dl.tensor_length == 0) {
        error = "download operation has zero length";
        return false;
    }
    if (dl.source_url.empty()) {
        error = "download operation missing source_url";
        return false;
    }
    if (!fetch_range(dl.source_url, dl.tensor_offset, dl.tensor_length, out)) {
        error = "range fetch failed";
        return false;
    }
    if (out.size() != dl.tensor_length) {
        out.resize(static_cast<size_t>(dl.tensor_length));
    }
    if (!checksum_matches(
                dl.checksum,
                out.data(),
                out.size(),
                dl.layer_index,
                dl.tensor_length)) {
        error = "checksum mismatch before store";
        return false;
    }
    return true;
}

executor_result http_range_download_executor::store_body(
        const install_operation & operation,
        layer_store & store,
        std::vector<uint8_t> body) {
    executor_result result;
    const download_operation & dl = operation.download;

    const uint64_t offset_end = dl.tensor_offset + dl.tensor_length;
    const bool is_blob_tensor = !dl.blob_id.empty() && !dl.tensor_name.empty();
    const bool stored = is_blob_tensor
            ? store.store_blob_tensor(
                    dl.blob_id,
                    dl.tensor_name,
                    body.data(),
                    body.size(),
                    dl.tensor_offset,
                    dl.checksum)
            : store.store_layer(
                    dl.layer_index,
                    body.data(),
                    body.size(),
                    dl.tensor_offset,
                    offset_end,
                    dl.checksum);
    if (!stored) {
        result.error = "failed to store layer blob";
        return result;
    }

    const bool verified = is_blob_tensor
            ? store.verify_blob_tensor(dl.blob_id, dl.tensor_name, dl.checksum)
            : store.verify_layer(dl.layer_index, dl.checksum);
    if (!verified) {
        if (is_blob_tensor) {
            store.remove_blob_tensor(dl.blob_id, dl.tensor_name);
        } else {
            store.remove_layer(dl.layer_index);
        }
        result.error = "post-store verification failed";
        return result;
    }

    result.success = true;
    return result;
}

executor_result http_range_download_executor::execute(
        const install_operation & operation,
        layer_store & store) {
    executor_result result;

    if (operation.action != install_action::download &&
            operation.action != install_action::repair) {
        result.error = "http_range executor supports DOWNLOAD/REPAIR only";
        return result;
    }

    std::vector<uint8_t> body;
    std::string fetch_error;
    if (!fetch_body(operation.download, body, fetch_error)) {
        result.error = fetch_error;
        return result;
    }
    return store_body(operation, store, std::move(body));
}