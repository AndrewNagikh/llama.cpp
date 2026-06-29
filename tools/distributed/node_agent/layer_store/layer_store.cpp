#include "layer_store.h"
#include "layer_special.h"

#include "architecture/semantic_blob.h"
#include "layer_checksum.h"
#include "layer_gguf_assembler.h"

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
    if (layer_index == layer_special::embedding) {
        return "special_embedding.bin";
    }
    if (layer_index == layer_special::output) {
        return "special_output.bin";
    }
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
    if (data == nullptr || len == 0) {
        return false;
    }
    if (layer_index < layer_special::output) {
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
        const int32_t layer_index = it.value().value("layer_index", -99);
        if (layer_index < layer_special::output) {
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

std::vector<layer_store::blob_tensor_info> layer_store::list_blob_tensors() const {
    std::vector<blob_tensor_info> blobs;
    const json checksums = load_checksums();
    if (!checksums.is_object()) {
        return blobs;
    }

    for (auto it = checksums.begin(); it != checksums.end(); ++it) {
        if (!it.value().is_object()) {
            continue;
        }
        const std::string blob_id     = it.value().value("blob_id", "");
        const std::string tensor_name = it.value().value("tensor_name", "");
        if (blob_id.empty() || tensor_name.empty()) {
            continue;
        }
        if (!has_blob_tensor(blob_id, tensor_name)) {
            continue;
        }
        blob_tensor_info info;
        info.blob_id     = blob_id;
        info.tensor_name = tensor_name;
        info.size_bytes  = it.value().value("size_bytes", static_cast<uint64_t>(0));
        info.checksum    = it.value().value("checksum", "");
        blobs.push_back(std::move(info));
    }

    return blobs;
}

bool layer_store::has_layer(const int32_t layer_index) const {
    std::error_code ec;
    return std::filesystem::exists(layer_path(layer_index), ec);
}

std::filesystem::path layer_store::blobs_dir() const {
    return model_root() / "blobs";
}

std::string layer_store::safe_tensor_filename(const std::string & tensor_name) {
    std::string safe = tensor_name;
    for (char & c : safe) {
        if (c == '/' || c == '\\') {
            c = '_';
        }
    }
    return safe + ".bin";
}

std::filesystem::path layer_store::blob_tensor_path(
        const std::string & blob_id,
        const std::string & tensor_name) const {
    return blobs_dir() / blob_id / safe_tensor_filename(tensor_name);
}

bool layer_store::store_blob_tensor(
        const std::string & blob_id,
        const std::string & tensor_name,
        const uint8_t * data,
        const size_t len,
        const uint64_t offset_begin,
        const std::string & checksum) {
    if (blob_id.empty() || tensor_name.empty() || data == nullptr || len == 0) {
        return false;
    }

    std::error_code ec;
    std::filesystem::create_directories(blobs_dir() / blob_id, ec);
    if (ec) {
        return false;
    }

    const auto path = blob_tensor_path(blob_id, tensor_name);
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
    const std::string key = blob_checksum_key(blob_id, tensor_name);
    checksums[key] = {
        { "blob_id", blob_id },
        { "tensor_name", tensor_name },
        { "offset_begin", offset_begin },
        { "offset_end", offset_begin + len },
        { "size_bytes", len },
        { "checksum", stored_checksum },
        { "path", (std::filesystem::path("blobs") / blob_id / safe_tensor_filename(tensor_name)).string() },
    };
    return save_checksums(checksums);
}

bool layer_store::load_blob_tensor(
        const std::string & blob_id,
        const std::string & tensor_name,
        std::vector<uint8_t> & out) const {
    out.clear();
    const auto path = blob_tensor_path(blob_id, tensor_name);
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

bool layer_store::has_blob_tensor(const std::string & blob_id, const std::string & tensor_name) const {
    std::error_code ec;
    return std::filesystem::exists(blob_tensor_path(blob_id, tensor_name), ec);
}

bool layer_store::has_semantic_blob(
        const std::string & blob_id,
        const std::vector<semantic_tensor_slot> & tensors) const {
    if (tensors.empty()) {
        return true;
    }
    for (const auto & slot : tensors) {
        if (!has_blob_tensor(blob_id, slot.name)) {
            return false;
        }
    }
    return true;
}

bool layer_store::verify_blob_tensor(
        const std::string & blob_id,
        const std::string & tensor_name,
        const std::string & expected_checksum) const {
    std::vector<uint8_t> data;
    if (!load_blob_tensor(blob_id, tensor_name, data)) {
        return false;
    }

    json checksums = load_checksums();
    const std::string key = blob_checksum_key(blob_id, tensor_name);
    uint64_t size_bytes = data.size();
    if (checksums.contains(key) && checksums[key].is_object()) {
        size_bytes = checksums[key].value("size_bytes", size_bytes);
    }

    return checksum_matches(
            expected_checksum,
            data.data(),
            data.size(),
            -1,
            size_bytes);
}

bool layer_store::remove_blob_tensor(const std::string & blob_id, const std::string & tensor_name) {
    std::error_code ec;
    std::filesystem::remove(blob_tensor_path(blob_id, tensor_name), ec);

    json checksums = load_checksums();
    checksums.erase(blob_checksum_key(blob_id, tensor_name));
    return save_checksums(checksums);
}

bool layer_store::save_metadata_blob(const std::vector<uint8_t> & data) {
    if (!ensure_dirs() || data.empty()) {
        return false;
    }
    std::ofstream out(model_root() / "metadata.bin", std::ios::binary | std::ios::trunc);
    if (!out) {
        return false;
    }
    out.write(reinterpret_cast<const char *>(data.data()), static_cast<std::streamsize>(data.size()));
    return static_cast<bool>(out);
}

std::optional<std::vector<uint8_t>> layer_store::load_metadata_blob() const {
    const auto path = model_root() / "metadata.bin";
    std::error_code ec;
    if (!std::filesystem::exists(path, ec)) {
        return std::nullopt;
    }
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        return std::nullopt;
    }
    in.seekg(0, std::ios::end);
    const auto size = in.tellg();
    if (size <= 0) {
        return std::nullopt;
    }
    in.seekg(0, std::ios::beg);
    std::vector<uint8_t> data(static_cast<size_t>(size));
    in.read(reinterpret_cast<char *>(data.data()), size);
    if (!in) {
        return std::nullopt;
    }
    return data;
}

std::optional<uint64_t> layer_store::metadata_bytes() const {
    const auto data = load_metadata_blob();
    if (!data.has_value()) {
        return std::nullopt;
    }
    return static_cast<uint64_t>(data->size());
}

bool layer_store::clear_model_storage(const bool keep_manifest) {
    const auto root = model_root();
    std::error_code ec;

    if (std::filesystem::exists(layers_dir(), ec)) {
        std::filesystem::remove_all(layers_dir(), ec);
    }
    if (std::filesystem::exists(blobs_dir(), ec)) {
        std::filesystem::remove_all(blobs_dir(), ec);
    }

    std::filesystem::remove(root / "checksums.json", ec);
    std::filesystem::remove(root / "metadata.bin", ec);
    std::filesystem::remove(root / "tokenizer.gguf", ec);
    std::filesystem::remove(root / "worker_entry.gguf", ec);
    std::filesystem::remove(root / "worker_middle.gguf", ec);
    std::filesystem::remove(root / "worker_final.gguf", ec);

    if (!keep_manifest) {
        std::filesystem::remove(root / "manifest.json", ec);
    }

    ensure_dirs();
    if (!keep_manifest) {
        return true;
    }
    return save_checksums(json::object());
}
