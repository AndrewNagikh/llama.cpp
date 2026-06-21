#pragma once

#include "dist_common.h"
#include "split_gen_common.h"
#include "split_tcp_wire.h"

#include "llama.h"

#include <atomic>
#include <mutex>
#include <string>
#include <thread>

struct dist_node_runtime {
    std::string model_path;
    std::string node_id;

    llama_model * model = nullptr;
    llama_context * ctx = nullptr;
    llama_sampler * smpl = nullptr;

    int32_t n_embd  = 0;
    int32_t n_layer = 0;
    int32_t n_vocab = 0;

    dist_configure_req cfg{};
    std::atomic<bool> configured{ false };
    std::atomic<bool> peer_ready{ false };
    std::atomic<bool> stop{ false };

    int ctrl_fd  = -1;
    int next_fd  = -1;
    int peer_fd  = -1;
    int listen_fd = -1;

    std::mutex llama_mu;
    std::thread peer_thread;

    bool load_model();
    void unload_model();

    bool configure(const dist_configure_req & req, std::string & err);
    void shutdown_pipeline();

    dist_gen_resp handle_entry_cmd(
            split_gen_cmd cmd,
            int32_t n_tokens,
            int32_t pos_start,
            const int32_t * tokens);

    void entry_ctrl_loop();
    void peer_loop();
};

bool dist_node_forward_to_next(
        dist_node_runtime & rt,
        int32_t n_tokens,
        int32_t pos_start,
        bool include_logits,
        double ms_local,
        dist_gen_resp & out);

bool dist_node_run_middle_step(dist_node_runtime & rt, int peer_fd);
bool dist_node_run_final_step(dist_node_runtime & rt, int peer_fd);
