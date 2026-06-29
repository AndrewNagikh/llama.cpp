#pragma once

#include "runtime_debug.h"
#include "runtime_state.h"
#include "hidden_transport.h"
#include "tensor_stats.h"

#include <cstdint>
#include <fstream>
#include <mutex>
#include <string>

// JSONL runtime trace recorder. Thread-safe append-only.

class trace_recorder {
public:
    trace_recorder();
    explicit trace_recorder(const std::string & role);

    void set_step(int32_t step);
    void set_phase(const std::string & phase);

    void emit_step_begin(
            int32_t step,
            const std::string & phase,
            int32_t token_id,
            int32_t position,
            int32_t seq_id = 0);

    void emit_hidden(
            int32_t step,
            const std::string & phase,
            const float * data,
            int32_t n_tokens,
            int32_t n_embd,
            const char * dump_tag = nullptr);

    void emit_logits(
            int32_t step,
            const std::string & phase,
            const float * logits,
            int32_t vocab_size,
            bool skip_sampler = false);

    void emit_position(
            int32_t step,
            const std::string & phase,
            int32_t position,
            int32_t past_tokens,
            int32_t seq_len,
            int32_t batch_size);

    void emit_kv(
            int32_t step,
            const std::string & phase,
            int32_t kv_entries,
            int32_t n_layer,
            int32_t seq_len);

    void emit_token_selected(
            int32_t step,
            const std::string & phase,
            int32_t token_id,
            int32_t position,
            bool via_argmax = false);

    void emit_runtime_state(const runtime_state_snapshot & snap);

    void emit_transport(const hidden_transport_trace & tr);

    const std::string & trace_path() const { return trace_path_; }

private:
    void ensure_open();
    void write_line(const std::string & json_line);
    std::string base_fields(
            const char * event,
            int32_t step,
            const std::string & phase) const;

    std::string role_;
    std::string trace_path_;
    mutable std::mutex mu_;
    std::ofstream out_;
    int32_t cur_step_ = -1;
    std::string cur_phase_;
};

trace_recorder * dist_debug_recorder();
void dist_debug_reset_recorder(const std::string & role);
