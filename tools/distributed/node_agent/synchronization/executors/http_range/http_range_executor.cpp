#include "http_range_executor.h"

#include "dist_common.h"
#include "node_agent/layer_store/layer_checksum.h"

#include "httplib.h"

#include <cstdio>
#include <fstream>
#include <vector>

namespace {

static bool read_file_range(
        const std::string & path,
        uint64_t offset,
        uint64_t length,
        std::vector<uint8_t> & out) {
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        return false;
    }
    in.seekg(static_cast<std::streamoff>(offset), std::ios::beg);
    if (!in) {
        return false;
    }
    out.resize(static_cast<size_t>(length));
    in.read(reinterpret_cast<char *>(out.data()), static_cast<std::streamsize>(length));
    return static_cast<bool>(in) || in.gcount() == static_cast<std::streamsize>(length);
}

static bool http_range_fetch(
        const std::string & url,
        uint64_t offset,
        uint64_t length,
        std::vector<uint8_t> & out) {
    if (length == 0) {
        return false;
    }

    httplib::Headers headers = {
        { "User-Agent", "distributed-llama-node-agent/0.1" },
        { "Accept", "*/*" },
    };

    const std::string token = dist_hf_token();
    if (!token.empty()) {
        headers.emplace("Authorization", "Bearer " + token);
    }

    const uint64_t end = offset + length - 1;
    char range_buf[80];
    snprintf(range_buf, sizeof(range_buf), "bytes=%llu-%llu",
            static_cast<unsigned long long>(offset),
            static_cast<unsigned long long>(end));
    headers.emplace("Range", range_buf);

    httplib::Client cli(url.c_str());
    cli.set_connection_timeout(30, 0);
    cli.set_read_timeout(300, 0);
    cli.set_follow_location(true);

    const auto res = cli.Get(url.c_str(), headers);
    if (!res || (res->status != 200 && res->status != 206)) {
        return false;
    }

    out.assign(res->body.begin(), res->body.end());
    return out.size() == length || (res->status == 200 && out.size() >= length);
}

static bool fetch_range(
        const std::string & source_url,
        uint64_t offset,
        uint64_t length,
        std::vector<uint8_t> & out) {
    if (source_url.rfind("file://", 0) == 0) {
        return read_file_range(source_url.substr(7), offset, length, out);
    }
    return http_range_fetch(source_url, offset, length, out);
}

} // namespace

executor_result http_range_download_executor::execute(
        const install_operation & operation,
        layer_store & store) {
    executor_result result;

    if (operation.action != install_action::download &&
            operation.action != install_action::repair) {
        result.error = "http_range executor supports DOWNLOAD/REPAIR only";
        return result;
    }

    const download_operation & dl = operation.download;
    if (dl.tensor_length == 0) {
        result.error = "download operation has zero length";
        return result;
    }
    if (dl.source_url.empty()) {
        result.error = "download operation missing source_url";
        return result;
    }

    std::vector<uint8_t> body;
    if (!fetch_range(dl.source_url, dl.tensor_offset, dl.tensor_length, body)) {
        result.error = "range fetch failed";
        return result;
    }
    if (body.size() != dl.tensor_length) {
        body.resize(static_cast<size_t>(dl.tensor_length));
    }

    if (!checksum_matches(
                dl.checksum,
                body.data(),
                body.size(),
                dl.layer_index,
                dl.tensor_length)) {
        result.error = "checksum mismatch before store";
        return result;
    }

    const uint64_t offset_end = dl.tensor_offset + dl.tensor_length;
    if (!store.store_layer(
                dl.layer_index,
                body.data(),
                body.size(),
                dl.tensor_offset,
                offset_end,
                dl.checksum)) {
        result.error = "failed to store layer blob";
        return result;
    }

    if (!store.verify_layer(dl.layer_index, dl.checksum)) {
        store.remove_layer(dl.layer_index);
        result.error = "post-store verification failed";
        return result;
    }

    result.success = true;
    return result;
}
