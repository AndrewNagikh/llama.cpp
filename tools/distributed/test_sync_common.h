#pragma once

#include "node_agent/layer_store/layer_store.h"
#include "test_install_common.h"

#include <filesystem>
#include <string>

#if !defined(_WIN32)
#include <unistd.h>
#endif

inline layer_store make_temp_layer_store(const std::string & model_id = "test-model") {
    const auto root = std::filesystem::temp_directory_path() /
            ("dist-layer-store-" + std::to_string(getpid()));
    std::filesystem::create_directories(root);
    return layer_store(root, model_id);
}

inline install_operation make_download_op_for_layer(
        int32_t layer_index,
        const std::string & node_id,
        const std::string & source_url,
        uint64_t offset,
        uint64_t length) {
    install_operation op;
    op.action      = install_action::download;
    op.node_id     = node_id;
    op.layer_index = layer_index;
    op.download.layer_index   = layer_index;
    op.download.node_id       = node_id;
    op.download.tensor_offset = offset;
    op.download.tensor_length = length;
    op.download.source_url    = source_url;
    op.download.checksum      = "manifest:layer:" + std::to_string(layer_index);
    return op;
}
