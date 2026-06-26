#include "layer_store.h"

#include "layer_checksum.h"

#include <algorithm>
#include <fstream>

using json = nlohmann::json;

layer_store::layer_store(std::filesystem::path root, std::string model_id)
        : root_(std::move(root)), model_id_(std::move(model_id)) {}

void layer_store::set_root(std::filesystem::path root, std::string model_id) {
    root_      = std::move(root);
    model_id_  = std::move(model_id);
}

std::filesystem::path layer_store::model_root() const {
    if (model_id_.empty()) {
        return root_;
    }
    return root_ / model_id_;
}

std::filesystem::path layer_store::layers_dir() const {
    return model_root() / "layers";
}

std::string layer_store::layer_filename(const int32_t layer_index) {
    char buf[16];
    snprintf(buf, sizeof(buf), "%03d.bin", layer_index);
    return buf;
}

std::filesystem::path layer_store::layer_path(const int32_t layer_index) const {
    return layers_dir() / layer_filename(layer_index);
}

bool layer_store::ensure_dirs() const {
    std::error_code ec;
    std::filesystem::create_directories(layers_dir(), ec);
    return !ec;
}

bool layer_store::save_checksums(const json & checksums) const {
    if (!ensure_dirs()) {
        return false;
    }
    std::ofstream out(model_root() / "checksums.json");
    if (!out) {
        return false;
    }
    out << checksums.dump(2);
    return true;
}

json layer_store::load_checksums() const {
    const auto path = model_root() / "checksums.json";
    std::error_code ec;
    if (!std::filesystem::exists(path, ec)) {
        return json::object();
    }
    std::ifstream in(path);
    if (!in) {
        return json::object();
    }
    try {
        json j;
        in >> j;
        return j.is_object() ? j : json::object();
    } catch (...) {
        return json::object();
    }
}

bool layer_store::save_manifest(const model_manifest & manifest) {
    if (!ensure_dirs()) {
        return false;
    }
    std::ofstream out(model_root() / "manifest.json");
    if (!out) {
        return false;
    }
    out << manifest.to_json().dump(2);
    return true;
}

std::optional<model_manifest> layer_store::load_manifest() const {
    const auto path = model_root() / "manifest.json";
    std::error_code ec;
    if (!std::filesystem::exists(path, ec)) {
        return std::nullopt;
    }
    std::ifstream in(path);
    if (!in) {
        return std::nullopt;
    }
    try {
        json j;
        in >> j;
        return model_manifest::from_json(j);
    } catch (...) {
        return std::nullopt;
    }
}

bool layer_store::store_layer(
        const int32_t layer_index,
        const uint8_t * data,
        const size_t len,
        const uint64_t offset_begin,
        const uint64_t offset_end,
        const std::string & checksum) {
    if (layer_index < 0 || data == nullptr || len == 0) {
        return false;
    }
    if (!ensure_dirs()) {
        return false;
    }

    const auto path = layer_path(layer_index);
    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    if (!out) {
        return false;
    }
    out.write(reinterpret_cast<const char *>(data), static_cast<std::streamsize>(len));
    if (!out) {
        return false;
    }

    const std::string stored_checksum = checksum.empty()
            ? compute_blob_checksum(data, len)
            : checksum;

    json checksums = load_checksums();
    const std::string key = std::to_string(layer_index);
    checksums[key] = {
        { "layer_index", layer_index },
        { "offset_begin", offset_begin },
        { "offset_end", offset_end },
        { "size_bytes", len },
        { "checksum", stored_checksum },
        { "path", layer_filename(layer_index) },
    };
    return save_checksums(checksums);
}

bool layer_store::load_layer(const int32_t layer_index, std::vector<uint8_t> & out) const {
    out.clear();
    const auto path = layer_path(layer_index);
    std::error_code ec;
    if (!std::filesystem::exists(path, ec)) {
        return false;
    }

    std::ifstream in(path, std::ios::binary);
    if (!in) {
        return false;
    }
    in.seekg(0, std::ios::end);
    const auto size = in.tellg();
    if (size <= 0) {
        return false;
    }
    in.seekg(0, std::ios::beg);
    out.resize(static_cast<size_t>(size));
    in.read(reinterpret_cast<char *>(out.data()), size);
    return static_cast<bool>(in);
}

bool layer_store::remove_layer(const int32_t layer_index) {
    std::error_code ec;
    std::filesystem::remove(layer_path(layer_index), ec);

    json checksums = load_checksums();
    checksums.erase(std::to_string(layer_index));
    return save_checksums(checksums);
}

std::optional<layer_blob> layer_store::get_layer(const int32_t layer_index) const {
    json checksums = load_checksums();
    const std::string key = std::to_string(layer_index);
    if (!checksums.contains(key) || !checksums[key].is_object()) {
        std::error_code ec;
        if (!std::filesystem::exists(layer_path(layer_index), ec)) {
            return std::nullopt;
        }
        layer_blob blob;
        blob.layer_index = layer_index;
        blob.path        = layer_path(layer_index);
        blob.size_bytes  = std::filesystem::file_size(blob.path, ec);
        return blob;
    }

    const json & entry = checksums[key];
    layer_blob blob;
    blob.layer_index  = entry.value("layer_index", layer_index);
    blob.offset_begin = entry.value("offset_begin", static_cast<uint64_t>(0));
    blob.offset_end   = entry.value("offset_end", static_cast<uint64_t>(0));
    blob.size_bytes   = entry.value("size_bytes", static_cast<uint64_t>(0));
    blob.checksum     = entry.value("checksum", "");
    blob.path         = layer_path(layer_index);
    return blob;
}

bool layer_store::verify_layer(const int32_t layer_index, const std::string & expected_checksum) const {
    const auto blob = get_layer(layer_index);
    if (!blob.has_value()) {
        return false;
    }

    std::vector<uint8_t> data;
    if (!load_layer(layer_index, data)) {
        return false;
    }
    if (blob->size_bytes > 0 && data.size() != blob->size_bytes) {
        return false;
    }

    return checksum_matches(
            expected_checksum,
            data.data(),
            data.size(),
            layer_index,
            blob->size_bytes > 0 ? blob->size_bytes : data.size());
}

std::vector<layer_blob> layer_store::list_layers() const {
    std::vector<layer_blob> layers;
    json checksums = load_checksums();
    if (!checksums.is_object()) {
        return layers;
    }

    for (auto it = checksums.begin(); it != checksums.end(); ++it) {
        if (!it.value().is_object()) {
            continue;
        }
        const int32_t layer_index = it.value().value("layer_index", -1);
        if (layer_index < 0) {
            continue;
        }
        if (auto blob = get_layer(layer_index)) {
            layers.push_back(*blob);
        }
    }

    std::sort(layers.begin(), layers.end(),
            [](const layer_blob & a, const layer_blob & b) {
                return a.layer_index < b.layer_index;
            });
    return layers;
}

bool layer_store::has_layer(const int32_t layer_index) const {
    std::error_code ec;
    return std::filesystem::exists(layer_path(layer_index), ec);
}
