#pragma once

#include "llama.h"

#include <string>
#include <vector>

struct context_params_snapshot {
    std::string label;
    int32_t n_ctx            = 0;
    int32_t n_batch          = 0;
    int32_t n_ubatch         = 0;
    bool causal_attn         = false;
    bool flash_attn          = false;
    bool auto_fa             = false;
    float rope_freq_base     = 0.0f;
    float rope_freq_scale    = 0.0f;
    uint32_t n_ctx_orig_yarn = 0;
    float yarn_ext_factor    = 0.0f;
    float yarn_attn_factor   = 0.0f;
    float yarn_beta_fast     = 0.0f;
    float yarn_beta_slow     = 0.0f;
    bool embeddings          = false;
    int32_t pooling_type     = 0;
    int32_t ctx_type         = 0;
    bool offload_kqv         = false;
    bool kv_unified          = false;
    int32_t kv_entries       = 0;
    int32_t kv_n_layer       = 0;
    int32_t layer_start      = 0;
    int32_t layer_end        = 0;
    int32_t n_threads        = 0;
    int32_t n_threads_batch  = 0;
    bool no_perf             = false;
    bool pipeline_parallel   = false;
    std::string backend;
};

struct context_params_field_diff {
    std::string field;
    std::string mono_value;
    std::string worker_value;
};

struct context_params_diff_report {
    context_params_snapshot mono{};
    context_params_snapshot worker{};
    std::vector<context_params_field_diff> diffs;
    bool match = false;
    std::string message;
};

context_params_snapshot capture_context_params(
        llama_context * ctx,
        const char * label);

context_params_diff_report compare_context_params(
        llama_context * mono_ctx,
        llama_context * worker_ctx);

std::string context_params_snapshot_json(const context_params_snapshot & snap);
std::string context_params_diff_json(const context_params_diff_report & report);
