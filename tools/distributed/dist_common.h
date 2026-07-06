#pragma once

#include "nlohmann/json.hpp"

#include <cstdint>
#include <string>
#include <vector>

enum dist_node_role : uint32_t {
    DIST_ROLE_UNCONFIGURED = 0,
    DIST_ROLE_ENTRY        = 1,
    DIST_ROLE_MIDDLE       = 2,
    DIST_ROLE_FINAL        = 3,
};

struct dist_node_memory {
    uint64_t total_ram_bytes  = 0;
    // Available RAM for budgeting (not merely "free" pages).
    uint64_t free_ram_bytes   = 0;
    uint64_t total_vram_bytes = 0;
    uint64_t free_vram_bytes  = 0;
    bool     has_gpu          = false;
};

struct dist_node_cpu {
    std::string cpu_name;
    int         physical_cores = 0;
    int         logical_cores  = 0;
    uint64_t    cache_l3_bytes = 0;
};

struct dist_node_system {
    std::string os;
    std::string arch;
};

struct dist_node_hardware {
    int32_t     cpu_threads   = 4;
    int32_t     ram_gb        = 0;
    std::string gpu_name      = "none";
    int32_t     gpu_vram_gb   = 0;
    std::string cpu_name;
    std::string backend       = "cpu";
    bool        has_gpu       = false;

    // Legacy compatibility: also expose byte-level memory.
    dist_node_memory memory;
    dist_node_cpu    cpu;
    dist_node_system system;
};

struct dist_node_capabilities {
    std::string gpu_backend   = "cpu";
    std::string gpu_name      = "none";
    int32_t     gpu_memory_mb = 0;
    int32_t     cpu_threads   = 4;
    std::vector<std::string> supported_arch = { "llama" };

    // Byte-level memory (Task 8).
    uint64_t total_ram_bytes  = 0;
    uint64_t free_ram_bytes   = 0;
    uint64_t total_vram_bytes = 0;
    uint64_t free_vram_bytes  = 0;
    bool     has_gpu          = false;

    std::string cpu_name;
    std::string os;
    std::string arch;
};

struct dist_node_performance {
    double score       = 1.0;
    double decode_tps  = 0.0;
    double prefill_tps = 0.0;
    double load_ms     = 0.0;
};

struct dist_node_info {
    std::string node_id;
    std::string host;
    int         http_port   = 0;
    int         n_layer     = 0;
    int         n_embd      = 0;
    int64_t     memory_total_mb = 0;
    int64_t     memory_free_mb  = 0;
    double      score       = 1.0;
    dist_node_performance performance;
    int64_t     last_seen   = 0;
    dist_node_hardware hardware;
    dist_node_capabilities caps;
    dist_node_memory memory;
    dist_node_cpu    cpu;
    dist_node_system system;
    bool online = true;
};

struct dist_pipeline_stage {
    std::string node_id;
    std::string host;
    int         http_port   = 0;
    int         ctrl_port   = 0;
    int         peer_port   = 0;
    int         layer_start = 0;
    int         layer_end   = 0;
    double      score       = 0.0;
    dist_node_role role = DIST_ROLE_UNCONFIGURED;
};

struct dist_configure_req {
    std::string    session_id;
    std::string    model_id;
    dist_node_role role        = DIST_ROLE_UNCONFIGURED;
    int            layer_start = 0;
    int            layer_end   = 0;
    int            ctrl_port   = 0;
    int            peer_port   = 0;
    std::string    next_host;
    int            next_port   = 0;
    std::string    peer_bind   = "0.0.0.0";
    bool           next_is_final = false;
    std::string    source_url;
    std::string    output_service_host;
    int            output_service_port = 0;
};

struct dist_gen_resp {
    bool        ok        = false;
    int32_t     token_id  = -1;
    std::string error;
    double      ms_a_compute  = 0.0;
    double      ms_ab_xfer     = 0.0;
    double      ms_b_compute   = 0.0;
    double      ms_bc_xfer     = 0.0;
    double      ms_c_compute   = 0.0;
    double      ms_c_sample    = 0.0;
};

static constexpr int DIST_MAX_NEW_TOKENS = 32;

bool dist_parse_host_port(const std::string & listen, std::string & host, int & port);
std::string dist_role_name(dist_node_role role);

// Legacy probes (kept for compatibility).
dist_node_capabilities dist_probe_capabilities();
void dist_probe_memory(int64_t & total_mb, int64_t & free_mb);

// Task 8: full node resource probe.
void dist_probe_node_memory(dist_node_memory & out);
void dist_probe_node_cpu(dist_node_cpu & out);
dist_node_system dist_probe_node_system();
double dist_bytes_to_gb(uint64_t bytes);

// Parse byte counts from JSON without int32 overflow (>2GB RAM/VRAM).
uint64_t dist_json_u64(const nlohmann::json & j, const char * key, uint64_t def = 0);

// Normalize backend name to planner device id: cpu / cuda / metal.
std::string dist_normalize_device(const std::string & backend, bool has_gpu);

// Hugging Face token for discovery/downloads (HF_TOKEN, HUGGINGFACE_HUB_TOKEN, ~/.cache/huggingface/token).
std::string dist_hf_token();

// Concurrent install operations per node (DIST_SYNC_PARALLELISM, default 4).
int dist_sync_parallelism();

int64_t dist_now_unix();
