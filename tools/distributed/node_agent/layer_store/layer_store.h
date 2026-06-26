#pragma once

#include "layer_blob.h"
#include "manifest_builder/manifest_builder.h"

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

private:
    std::filesystem::path root_;
    std::string           model_id_;

    bool ensure_dirs() const;
    bool save_checksums(const nlohmann::json & checksums) const;
    nlohmann::json load_checksums() const;
    static std::string layer_filename(int32_t layer_index);
};
