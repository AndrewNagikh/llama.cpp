#pragma once

#include <cstdint>
#include <string>

// Task 12 — passive distributed runtime performance profiler.
// Enabled with DIST_PERF_TRACE=1. Uses steady_clock microseconds only.

enum class perf_category {
    COMPUTE,
    WAIT,
    NETWORK,
    SERIALIZATION,
    SAMPLING,
    IDLE,
    INSTALL_REUSE,
    SESSION,
    TTFT,
    GPU,
    UNKNOWN,
};

const char * perf_category_name(perf_category category);

struct perf_trace_config {
    bool        enabled    = false;
    std::string trace_dir;
    std::string node_id;
    std::string component;
};

bool perf_trace_enabled();
void perf_trace_load_config();
const perf_trace_config & perf_trace_config_get();

void perf_trace_set_node_id(const std::string & node_id);
void perf_trace_set_component(const std::string & component);

uint64_t perf_now_us();
int64_t  perf_trace_epoch_us();

// Active generate context (shared via active_context.json in trace_dir).
void perf_trace_set_context(const std::string & trace_id, const std::string & phase, int32_t token_idx);
void perf_trace_refresh_context();
bool perf_trace_get_context(std::string & trace_id, std::string & phase, int32_t & token_idx);
bool perf_trace_has_active_context();

void perf_trace_begin_generate(const std::string & trace_id, const std::string & subdir = "decode");
void perf_trace_end_generate();

void perf_trace_begin_install(const std::string & trace_id, const std::string & subdir = "install");
void perf_trace_end_install();

void perf_emit_install_span(
        const char * event,
        const char * sub,
        const char * blob_id,
        const char * node_id,
        uint64_t bytes,
        int64_t dur_us);

void perf_emit_install_instant(
        const char * event,
        const char * sub,
        const char * blob_id,
        const char * node_id,
        uint64_t bytes,
        const char * attrs_json = nullptr);

class perf_install_span {
public:
    perf_install_span(
            const char * event,
            const char * sub,
            const char * blob_id = nullptr,
            const char * node_id = nullptr);
    ~perf_install_span();

    void set_bytes(uint64_t bytes) { bytes_ = bytes; }

private:
    const char * event_;
    const char * sub_;
    const char * blob_id_;
    const char * node_id_;
    uint64_t     bytes_  = 0;
    int64_t      t0_us_  = 0;
    bool         active_ = false;
};

void perf_emit_span(
        const char * event,
        perf_category category,
        const char * stage,
        int32_t token_idx,
        int64_t dur_us,
        const char * attrs_json = nullptr);

void perf_emit_instant(
        const char * event,
        perf_category category,
        const char * stage,
        int32_t token_idx,
        const char * attrs_json = nullptr);

void perf_emit_hidden_transfer(
        const char * stage,
        int32_t token_idx,
        const char * link,
        int32_t payload_bytes,
        int64_t serialize_us,
        int64_t send_us,
        int64_t receive_us,
        int64_t deserialize_us);

void perf_emit_queue_depth(const char * stage, int32_t token_idx, int32_t depth);

void perf_emit_gpu_sample(
        const char * backend,
        float gpu_util_pct,
        float gpu_mem_used_mb,
        const char * attrs_json = nullptr);

class perf_span {
public:
    perf_span(const char * begin_event, const char * end_event, perf_category category, const char * stage);
    ~perf_span();

    void set_token_idx(int32_t token_idx) { token_idx_ = token_idx; }

private:
    const char *    begin_event_;
    const char *    end_event_;
    perf_category   category_;
    const char *    stage_;
    int32_t         token_idx_ = -1;
    int64_t         t0_us_     = 0;
    bool            active_    = false;
};

const char * perf_trace_output_dir();

std::string perf_make_trace_id(int seq);
std::string perf_make_install_trace_id(const std::string & run_id, const std::string & model_id);
std::string perf_make_session_trace_id(const std::string & session_id);

void perf_trace_begin_session(const std::string & trace_id, const std::string & subdir = "session");
void perf_trace_end_session();

void perf_emit_session_span(
        const char * event,
        const char * node_id,
        const char * role,
        int64_t dur_us,
        const char * attrs_json = nullptr);

void perf_emit_session_instant(
        const char * event,
        const char * node_id,
        const char * role,
        const char * attrs_json = nullptr);

class perf_session_span {
public:
    perf_session_span(const char * event, const char * node_id, const char * role);
    ~perf_session_span();

private:
    const char * event_;
    const char * node_id_;
    const char * role_;
    int64_t      t0_us_  = 0;
    bool         active_ = false;
};

void perf_trace_begin_ttft(const std::string & trace_id, const std::string & subdir = "ttft");
void perf_trace_end_ttft();

void perf_emit_ttft_span(
        const char * event,
        const char * stage,
        int64_t dur_us,
        const char * attrs_json = nullptr);

void perf_emit_ttft_instant(
        const char * event,
        const char * stage,
        const char * attrs_json = nullptr);

class perf_ttft_span {
public:
    perf_ttft_span(const char * event, const char * stage);
    ~perf_ttft_span();

private:
    const char * event_;
    const char * stage_;
    int64_t      t0_us_  = 0;
    bool         active_ = false;
};
