#pragma once

#include "context_params_verify.h"
#include "decode_graph_verify.h"
#include "runtime_cache_verify.h"
#include "runtime_debug/tensor_stats.h"

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

struct final_runtime_config {
    std::string mono_path;
    std::string worker_final_path;
    std::string prompt;
    int32_t layer_boundary = -1; // mono [0,boundary), worker [boundary,n_layer)
    int32_t max_decode_steps = 8;
    std::string output_dir;
    bool save_hidden_bin     = true;
};

struct logits_snapshot {
    std::vector<float> data;
    tensor_stats stats;
    int32_t argmax     = -1;
    float entropy      = 0.0f;
    std::vector<std::pair<int32_t, float>> top10;
};

struct hidden_consumption_diag {
    std::string sha256_before;
    std::string sha256_after;
    size_t hidden_bytes     = 0;
    size_t stride           = sizeof(float);
    const char * dtype      = "f32";
    uintptr_t hidden_ptr_before = 0;
    uintptr_t hidden_ptr_after  = 0;
    bool hidden_modified    = false;
    bool api_roundtrip_ok   = false;
    std::string message;
};

struct kv_ownership_diag {
    uintptr_t kv_ptr     = 0;
    int32_t kv_entries   = 0;
    int32_t kv_n_layer   = 0;
    int32_t seq_count    = 1;
    int32_t n_past       = 0;
    std::string phase;
};

struct decode_step_equivalence {
    int32_t step         = -1;
    std::string phase;
    logits_snapshot mono;
    logits_snapshot worker;
    parity_metrics parity;
    bool pass            = false;
    std::string message;
};

struct final_runtime_report {
    bool all_pass              = false;
    bool prefill_logits_pass   = false;
    int32_t layer_boundary     = 0;
    int32_t n_tokens           = 0;
    int32_t n_embd             = 0;
    std::string hidden_sha256;
    std::string hidden_bin_path;
    logits_snapshot mono_prefill_logits;
    logits_snapshot worker_prefill_logits;
    parity_metrics prefill_parity;
    context_params_diff_report context_params;
    decode_graph_verify_result decode_graph;
    decode_graph_verify_result prefill_graph;
    runtime_cache_diag mono_cache;
    runtime_cache_diag worker_cache;
    hidden_consumption_diag hidden_consumption;
    kv_ownership_diag kv_after_prefill;
    std::vector<decode_step_equivalence> decode_steps;
    int32_t first_fail_decode_step = -1;
    std::string root_cause_field;
    std::string message;
};

final_runtime_report verify_final_runtime(const final_runtime_config & cfg);

std::string final_runtime_report_json(const final_runtime_report & report);

// Logits-only comparison (hidden already injected externally).
struct final_logits_config {
    std::string mono_path;
    std::string worker_final_path;
    std::string hidden_bin_path;
    std::string prompt;
    int32_t layer_boundary = -1;
};

struct final_logits_report {
    bool pass = false;
    logits_snapshot mono;
    logits_snapshot worker;
    parity_metrics parity;
    std::string message;
};

final_logits_report verify_final_logits(const final_logits_config & cfg);
std::string final_logits_report_json(const final_logits_report & report);

logits_snapshot capture_logits_snapshot(const float * logits, int32_t n_vocab);
