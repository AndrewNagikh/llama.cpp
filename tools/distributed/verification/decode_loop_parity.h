#pragma once

#include "runtime_debug/tensor_stats.h"

#include <string>
#include <vector>

enum class divergence_kind {
    none,
    token,
    position,
    hidden,
    logits,
    kv,
    selected_token,
    missing_event,
};

struct divergence_report {
    divergence_kind kind         = divergence_kind::none;
    int32_t step                 = -1;
    std::string phase;
    std::string worker;
    std::string mono_source;
    std::string dist_source;
    std::string field;
    std::string message;
    parity_metrics metrics{};
    std::string root_cause_hint;
};

struct trace_event {
    std::string event;
    int32_t step = -1;
    std::string phase;
    std::string worker;
    std::string session;
    std::string node;
    int32_t token = -1;
    int32_t position = -1;
    int32_t seq_id = 0;
    int32_t n_tokens = 0;
    int32_t n_embd = 0;
    int32_t vocab_size = 0;
    int32_t argmax = -1;
    float argmax_score = 0.0f;
    float entropy = 0.0f;
    int32_t kv_entries = 0;
    int32_t seq_len = 0;
    int32_t past_tokens = 0;
    tensor_stats hidden_stats{};
    tensor_stats logits_stats{};
    std::string sha256;
    bool has_hidden = false;
    bool has_logits = false;
};

struct parsed_trace {
    std::string path;
    std::string worker;
    std::vector<trace_event> events;
};

parsed_trace parse_trace_jsonl(const std::string & path);

divergence_report find_first_divergence(
        const parsed_trace & mono,
        const parsed_trace & dist_entry,
        const parsed_trace & dist_final,
        const parsed_trace & dist_middle = {});

std::string divergence_kind_name(divergence_kind kind);
