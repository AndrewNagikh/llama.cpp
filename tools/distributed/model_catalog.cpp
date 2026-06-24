#include "model_catalog.h"

#include "nlohmann/json.hpp"

#include <fstream>
#include <sstream>
#include <chrono>
#include <filesystem>
#include <cstdlib>

using json = nlohmann::json;

// Simple JSON-like parser/writer for catalog data
// Note: Using minimal JSON handling to avoid external dependencies

static std::string escape_json_string(const std::string & str) {
    std::string result;
    for (char c : str) {
        switch (c) {
            case '"':  result += "\\\""; break;
            case '\\': result += "\\\\"; break;
            case '\n': result += "\\n"; break;
            case '\r': result += "\\r"; break;
            case '\t': result += "\\t"; break;
            default: result += c; break;
        }
    }
    return result;
}

static uint64_t current_time_ms() {
    auto now = std::chrono::system_clock::now();
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(now.time_since_epoch());
    return static_cast<uint64_t>(ms.count());
}

// Model Catalog Implementation

bool model_catalog::load_catalog(const std::string & catalog_path) {
    std::ifstream file(catalog_path);
    if (!file.is_open()) {
        // Initialize with defaults if file doesn't exist
        init_default_catalog();
        return save_catalog(catalog_path);
    }
    
    // Simple JSON parsing (just for basic structure)
    std::string line, content;
    while (std::getline(file, line)) {
        content += line;
    }
    
    // For now, initialize with defaults (proper JSON parsing can be added later)
    init_default_catalog();
    return true;
}

bool model_catalog::save_catalog(const std::string & catalog_path) const {
    const std::filesystem::path path(catalog_path);
    if (path.has_parent_path()) {
        std::error_code ec;
        std::filesystem::create_directories(path.parent_path(), ec);
    }

    std::ofstream file(catalog_path);
    if (!file.is_open()) {
        return false;
    }
    
    file << "{\n";
    file << "  \"models\": [\n";
    
    for (size_t i = 0; i < models_.size(); ++i) {
        const auto & model = models_[i];
        if (i > 0) file << ",\n";
        
        file << "    {\n";
        file << "      \"id\": \"" << escape_json_string(model.id) << "\",\n";
        file << "      \"display_name\": \"" << escape_json_string(model.display_name) << "\",\n";
        file << "      \"n_layers\": " << model.n_layers << ",\n";
        file << "      \"n_embd\": " << model.n_embd << ",\n";
        file << "      \"size_gb\": " << model.size_gb << ",\n";
        file << "      \"source\": {\n";
        file << "        \"repo\": \"" << escape_json_string(model.source.repo) << "\",\n";
        file << "        \"file\": \"" << escape_json_string(model.source.file) << "\"\n";
        file << "      }\n";
        file << "    }";
    }
    
    file << "\n  ]\n";
    file << "}\n";
    return true;
}

std::vector<model_info> model_catalog::get_models() const {
    return models_;
}

model_info * model_catalog::find_model(const std::string & model_id) {
    for (auto & model : models_) {
        if (model.id == model_id) {
            return &model;
        }
    }
    return nullptr;
}

const model_info * model_catalog::find_model(const std::string & model_id) const {
    for (const auto & model : models_) {
        if (model.id == model_id) {
            return &model;
        }
    }
    return nullptr;
}

bool model_catalog::add_model(const model_info & model) {
    if (find_model(model.id) != nullptr) {
        return false; // Already exists
    }
    models_.push_back(model);
    return true;
}

std::string model_catalog::create_install_job(const std::string & model_id) {
    std::string job_id = generate_job_id();
    
    install_job job;
    job.job_id = job_id;
    job.model_id = model_id;
    job.status = install_status::unknown;
    job.started_ms = current_time_ms();
    
    install_jobs_[job_id] = job;
    return job_id;
}

install_job * model_catalog::get_install_job(const std::string & job_id) {
    auto it = install_jobs_.find(job_id);
    return (it != install_jobs_.end()) ? &it->second : nullptr;
}

void model_catalog::update_job_status(const std::string & job_id, install_status status, 
                                     const std::string & error_msg, double progress) {
    auto * job = get_install_job(job_id);
    if (job) {
        job->status = status;
        job->error_msg = error_msg;
        job->progress = progress;
    }
}

void model_catalog::complete_job(const std::string & job_id) {
    auto * job = get_install_job(job_id);
    if (job) {
        job->status = install_status::ready;
        job->progress = 1.0;
        job->completed_ms = current_time_ms();
    }
}

std::vector<model_placement> model_catalog::plan_placement(const std::string & model_id,
                                                          const std::vector<std::string> & node_ids,
                                                          const std::vector<uint64_t> & node_memory) const {
    std::vector<model_placement> placements;
    
    const auto * model = find_model(model_id);
    if (!model || node_ids.empty()) {
        return placements;
    }
    
    int32_t n_layers = model->n_layers;
    size_t n_nodes = node_ids.size();
    
    // Simple equal distribution for now (similar to layer_planner)
    // TODO: Consider node memory and model layer costs
    
    int32_t base_layers = n_layers / static_cast<int32_t>(n_nodes);
    int32_t extra_layers = n_layers % static_cast<int32_t>(n_nodes);
    
    int32_t current_layer = 0;
    for (size_t i = 0; i < n_nodes; ++i) {
        model_placement placement;
        placement.node_id = node_ids[i];
        placement.layer_start = current_layer;
        
        int32_t layers_for_this_node = base_layers + (i < static_cast<size_t>(extra_layers) ? 1 : 0);
        placement.layer_end = current_layer + layers_for_this_node;
        
        // Rough memory estimation (will be refined in future tasks)
        placement.estimated_memory = static_cast<uint64_t>(model->size_gb * 1024 * 1024 * 1024 * layers_for_this_node / n_layers);
        
        placements.push_back(placement);
        current_layer = placement.layer_end;
    }
    
    return placements;
}

void model_catalog::init_default_catalog() {
    models_.clear();
    
    // Add default model (Llama 3.2 1B Instruct Q4_K_M)
    model_info llama_1b;
    llama_1b.id = "llama-3.2-1b";
    llama_1b.display_name = "Llama 3.2 1B Instruct Q4_K_M";
    llama_1b.n_layers = 16;
    llama_1b.n_embd = 2048;
    llama_1b.size_gb = 0.8;
    llama_1b.source.repo = "hugging-quants/Llama-3.2-1B-Instruct-Q4_K_M-GGUF";
    llama_1b.source.file = "llama-3.2-1b-instruct-q4_k_m.gguf";
    
    models_.push_back(llama_1b);
}

std::string model_catalog::generate_job_id() const {
    auto now = current_time_ms();
    std::ostringstream oss;
    oss << "install-" << now;
    return oss.str();
}

// Model Store Implementation

bool model_store::load_state(const std::string & state_path) {
    ensure_models_dir();

    std::ifstream file(state_path);
    if (!file.is_open()) {
        return true; // Empty state is OK
    }

    try {
        json state = json::parse(file);
        local_models_.clear();
        for (const auto & item : state.value("models", json::array())) {
            installed_model model;
            model.model_id = item.value("model_id", "");
            model.local_path = item.value("local_path", "");
            model.size_bytes = item.value("size_bytes", static_cast<uint64_t>(0));
            model.ready = item.value("ready", false);
            model.installed_ms = item.value("installed_ms", static_cast<uint64_t>(0));
            if (!model.model_id.empty() && !model.local_path.empty()) {
                local_models_.push_back(model);
            }
        }
    } catch (...) {
        return false;
    }

    return true;
}

bool model_store::save_state(const std::string & state_path) const {
    ensure_models_dir();
    
    std::ofstream file(state_path);
    if (!file.is_open()) {
        return false;
    }
    
    file << "{\n";
    file << "  \"models\": [\n";
    
    for (size_t i = 0; i < local_models_.size(); ++i) {
        const auto & model = local_models_[i];
        if (i > 0) file << ",\n";
        
        file << "    {\n";
        file << "      \"model_id\": \"" << escape_json_string(model.model_id) << "\",\n";
        file << "      \"local_path\": \"" << escape_json_string(model.local_path) << "\",\n";
        file << "      \"size_bytes\": " << model.size_bytes << ",\n";
        file << "      \"ready\": " << (model.ready ? "true" : "false") << ",\n";
        file << "      \"installed_ms\": " << model.installed_ms << "\n";
        file << "    }";
    }
    
    file << "\n  ]\n";
    file << "}\n";
    return true;
}

std::vector<installed_model> model_store::get_local_models() const {
    return local_models_;
}

installed_model * model_store::find_model(const std::string & model_id) {
    for (auto & model : local_models_) {
        if (model.model_id == model_id) {
            return &model;
        }
    }
    return nullptr;
}

const installed_model * model_store::find_model(const std::string & model_id) const {
    for (const auto & model : local_models_) {
        if (model.model_id == model_id) {
            return &model;
        }
    }
    return nullptr;
}

bool model_store::add_model(const installed_model & model) {
    auto * existing = find_model(model.model_id);
    if (existing) {
        *existing = model; // Update existing
    } else {
        local_models_.push_back(model);
    }
    return true;
}

bool model_store::remove_model(const std::string & model_id) {
    for (auto it = local_models_.begin(); it != local_models_.end(); ++it) {
        if (it->model_id == model_id) {
            local_models_.erase(it);
            return true;
        }
    }
    return false;
}

void model_store::set_models_dir(const std::string & dir) {
    models_dir_ = dir;
}

std::string model_store::get_models_dir() const {
    if (!models_dir_.empty()) {
        return models_dir_;
    }
    
    const char * home = std::getenv("HOME");
    if (!home) {
        return "/tmp/.distributed-llm/models";
    }
    
    return std::string(home) + "/.distributed-llm/models";
}

std::string model_store::get_model_path(const std::string & model_id) const {
    return get_models_dir() + "/" + model_id + "/model.gguf";
}

void model_store::ensure_models_dir() const {
    std::string dir = get_models_dir();
    std::filesystem::create_directories(dir);
}