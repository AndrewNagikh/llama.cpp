#include "semantic_blob.h"

#include <cstdio>

const semantic_blob * find_blob(const std::vector<semantic_blob> & blobs, const std::string & id) {
    for (const auto & blob : blobs) {
        if (blob.id == id) {
            return &blob;
        }
    }
    return nullptr;
}

semantic_blob * find_blob_mut(std::vector<semantic_blob> & blobs, const std::string & id) {
    for (auto & blob : blobs) {
        if (blob.id == id) {
            return &blob;
        }
    }
    return nullptr;
}

std::string blob_checksum_key(const std::string & blob_id, const std::string & tensor_name) {
    return "blob:" + blob_id + ":" + tensor_name;
}

std::string layer_blob_id(const int32_t layer_index) {
    char buf[32];
    snprintf(buf, sizeof(buf), "layer:%d", layer_index);
    return buf;
}
