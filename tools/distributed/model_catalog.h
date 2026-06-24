#pragma once

#include <string>
#include <vector>
#include <map>
#include <cstdint>

// Model metadata for catalog
struct model_info {
    std::string id;                  // "llama-3.2-1b"
    std::string display_name;        // "Llama 3.2 1B Instruct Q4_K_M" 
    int32_t n_layers = 0;
    int32_t n_embd = 0;
    double size_gb = 0.0;
    
    // Source information for downloading
    struct {
        std::string repo;            // "hugging-quants/Llama-3.2-1B-Instruct-Q4_K_M-GGUF"
        std::string file;            // "llama-3.2-1b-instruct-q4_k_m.gguf"
    } source;
};

// Installation status
enum class install_status {
    unknown,
    downloading,
    ready,
    error
};

// Model installation job
struct install_job {
    std::string job_id;
    std::string model_id;
    install_status status = install_status::unknown;
    std::string error_msg;
    double progress = 0.0;           // 0.0 to 1.0
    uint64_t started_ms = 0;
    uint64_t completed_ms = 0;
};

// Per-node model storage info
struct installed_model {
    std::string model_id;
    std::string local_path;          // Full path to .gguf file
    uint64_t size_bytes = 0;
    bool ready = false;
    uint64_t installed_ms = 0;       // Timestamp when installation completed
};

// Future layer placement (preparation for Task 8/9)
struct model_placement {
    std::string node_id;
    int32_t layer_start = 0;
    int32_t layer_end = -1;          // -1 = all remaining layers
    uint64_t estimated_memory = 0;   // Bytes
};

class model_catalog {
public:
    model_catalog() = default;
    
    // Load catalog from file or initialize with defaults
    bool load_catalog(const std::string & catalog_path);
    bool save_catalog(const std::string & catalog_path) const;
    
    // Catalog management
    std::vector<model_info> get_models() const;
    model_info * find_model(const std::string & model_id);
    const model_info * find_model(const std::string & model_id) const;
    bool add_model(const model_info & model);
    
    // Installation job management
    std::string create_install_job(const std::string & model_id);
    install_job * get_install_job(const std::string & job_id);
    void update_job_status(const std::string & job_id, install_status status, 
                          const std::string & error_msg = "", double progress = 0.0);
    void complete_job(const std::string & job_id);
    
    // Placement planning (for future use)
    std::vector<model_placement> plan_placement(const std::string & model_id,
                                               const std::vector<std::string> & node_ids,
                                               const std::vector<uint64_t> & node_memory) const;
    
private:
    std::vector<model_info> models_;
    std::map<std::string, install_job> install_jobs_;
    
    void init_default_catalog();
    std::string generate_job_id() const;
};

// Node-side model store
class model_store {
public:
    model_store() = default;
    
    // Load/save local model state
    bool load_state(const std::string & state_path);
    bool save_state(const std::string & state_path) const;
    
    // Model management
    std::vector<installed_model> get_local_models() const;
    installed_model * find_model(const std::string & model_id);
    const installed_model * find_model(const std::string & model_id) const;
    
    bool add_model(const installed_model & model);
    bool remove_model(const std::string & model_id);
    
    // Installation helpers
    std::string get_models_dir() const;          // ~/.distributed-llm/models/
    std::string get_model_path(const std::string & model_id) const;
    
private:
    std::vector<installed_model> local_models_;
    std::string models_dir_;
    
    void ensure_models_dir() const;
};