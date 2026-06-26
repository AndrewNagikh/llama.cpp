#include "registry_persistence.h"

#include <filesystem>
#include <fstream>

using json = nlohmann::json;

namespace {

std::filesystem::path registry_file_path(const std::string & models_dir) {
    return std::filesystem::path(models_dir) / ".orchestrator" / "registry.json";
}

} // namespace

bool registry_persistence_load(const std::string & models_dir, cluster_model_registry & registry) {
    if (models_dir.empty()) {
        return false;
    }

    const auto path = registry_file_path(models_dir);
    std::error_code ec;
    if (!std::filesystem::exists(path, ec)) {
        return false;
    }

    json root;
    try {
        std::ifstream in(path);
        if (!in) {
            return false;
        }
        in >> root;
    } catch (...) {
        return false;
    }

    if (!root.contains("models") || !root["models"].is_array()) {
        return false;
    }

    int loaded = 0;
    for (const auto & item : root["models"]) {
        dist_model_record record = dist_model_record_from_json(item);
        if (record.model_id.empty()) {
            continue;
        }
        registry.add_or_update(record);
        ++loaded;
    }

    return loaded > 0;
}

bool registry_persistence_save(const std::string & models_dir, const cluster_model_registry & registry) {
    if (models_dir.empty()) {
        return false;
    }

    const auto path = registry_file_path(models_dir);
    std::error_code ec;
    std::filesystem::create_directories(path.parent_path(), ec);

    json models = json::array();
    for (const auto & record : registry.list()) {
        models.push_back(record.to_json());
    }

    const json root = {
        { "version", 1 },
        { "models", models },
    };

    std::ofstream out(path, std::ios::trunc);
    if (!out) {
        return false;
    }
    out << root.dump(2);
    return static_cast<bool>(out);
}
