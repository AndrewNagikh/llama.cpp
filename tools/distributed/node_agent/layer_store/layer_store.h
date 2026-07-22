#pragma once

#include "layer_blob.h"
#include "manifest_builder/manifest_builder.h"

#include "architecture/semantic_blob.h"

#include "nlohmann/json.hpp"

#include <filesystem>
#include <optional>
#include <string>
#include <vector>

// Local per-model layer storage (Task 9.7).
//
//   {root}/{model_id}/
//     manifest.json
//     checksums.json
//     layers/000.bin ...

class layer_store {
public:
    layer_store() = default;
    explicit layer_store(std::filesystem::path root, std::string model_id = "");

    void set_root(std::filesystem::path root, std::string model_id);

    const std::string & model_id() const { return model_id_; }

    std::filesystem::path model_root() const;
    std::filesystem::path layers_dir() const;
    std::filesystem::path layer_path(int32_t layer_index) const;

    bool save_manifest(const model_manifest & manifest);
    std::optional<model_manifest> load_manifest() const;

    bool store_layer(
            int32_t layer_index,
            const uint8_t * data,
            size_t len,
            uint64_t offset_begin,
            uint64_t offset_end,
            const std::string & checksum);

    bool load_layer(int32_t layer_index, std::vector<uint8_t> & out) const;
    bool remove_layer(int32_t layer_index);
    bool verify_layer(int32_t layer_index, const std::string & expected_checksum) const;
    std::vector<layer_blob> list_layers() const;
    bool has_layer(int32_t layer_index) const;
    std::optional<layer_blob> get_layer(int32_t layer_index) const;

    std::filesystem::path blobs_dir() const;
    std::filesystem::path blob_tensor_path(
            const std::string & blob_id,
            const std::string & tensor_name) const;

    bool store_blob_tensor(
            const std::string & blob_id,
            const std::string & tensor_name,
            const uint8_t * data,
            size_t len,
            uint64_t offset_begin,
            const std::string & checksum);

    bool load_blob_tensor(
            const std::string & blob_id,
            const std::string & tensor_name,
            std::vector<uint8_t> & out) const;

    bool has_blob_tensor(const std::string & blob_id, const std::string & tensor_name) const;
    bool has_semantic_blob(
            const std::string & blob_id,
            const std::vector<semantic_tensor_slot> & tensors) const;
    bool verify_blob_tensor(
            const std::string & blob_id,
            const std::string & tensor_name,
            const std::string & expected_checksum) const;
    bool remove_blob_tensor(const std::string & blob_id, const std::string & tensor_name);

    struct blob_tensor_info {
        std::string blob_id;
        std::string tensor_name;
        uint64_t    size_bytes = 0;
        std::string checksum;
    };
    std::vector<blob_tensor_info> list_blob_tensors() const;

    bool save_metadata_blob(const std::vector<uint8_t> & data);
    std::optional<std::vector<uint8_t>> load_metadata_blob() const;
    std::optional<uint64_t> metadata_bytes() const;

    // Remove all layer/blob bytes and checksums. Optionally keep manifest + metadata.
    bool clear_model_storage(bool keep_manifest = false);

private:
    std::filesystem::path root_;
    std::string           model_id_;

    bool ensure_dirs() const;
    bool save_checksums(const nlohmann::json & checksums) const;
    nlohmann::json load_checksums() const;
    static std::string layer_filename(int32_t layer_index);
    static std::string safe_path_component(const std::string & name);
    static std::string safe_tensor_filename(const std::string & tensor_name);
};
