#pragma once

#include <cstdint>
#include <string>
#include <vector>

enum dist_node_role : uint32_t {
    DIST_ROLE_UNCONFIGURED = 0,
    DIST_ROLE_ENTRY        = 1,
    DIST_ROLE_MIDDLE       = 2,
    DIST_ROLE_FINAL        = 3,
};

struct dist_node_capabilities {
    std::string gpu_backend   = "cpu";
    std::string gpu_name      = "none";
    int32_t     gpu_memory_mb = 0;
    int32_t     cpu_threads   = 4;
    std::vector<std::string> supported_arch = { "llama" };
};

struct dist_node_info {
    std::string node_id;
    std::string host;
    int         http_port   = 0;
    int         n_layer     = 0;
    int         n_embd      = 0;
    int64_t     memory_total_mb = 0;
    int64_t     memory_free_mb  = 0;
    dist_node_capabilities caps;
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
    dist_node_role role = DIST_ROLE_UNCONFIGURED;
};

struct dist_configure_req {
    dist_node_role role        = DIST_ROLE_UNCONFIGURED;
    int            layer_start = 0;
    int            layer_end   = 0;
    int            ctrl_port   = 0;
    int            peer_port   = 0;
    std::string    next_host;
    int            next_port   = 0;
    std::string    peer_bind   = "0.0.0.0";
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

static constexpr int DIST_LAYOUT_A_END  = 5;
static constexpr int DIST_LAYOUT_B_START = 5;
static constexpr int DIST_LAYOUT_B_END   = 10;
static constexpr int DIST_LAYOUT_C_START = 10;

bool dist_parse_host_port(const std::string & listen, std::string & host, int & port);
std::string dist_role_name(dist_node_role role);
dist_node_capabilities dist_probe_capabilities();
void dist_probe_memory(int64_t & total_mb, int64_t & free_mb);
