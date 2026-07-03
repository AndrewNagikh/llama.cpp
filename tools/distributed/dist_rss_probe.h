#pragma once

#include <cstdint>
#include <string>

// Process resident set size (best-effort; 0 if unavailable).
uint64_t dist_process_rss_bytes();

// Baseline captured once at orchestrator startup (after registry restore).
void dist_rss_set_baseline(uint64_t bytes);
uint64_t dist_rss_baseline_bytes();

// Log RSS for a named control-plane stage. detail is optional (model id, etc.).
void dist_rss_log_stage(const char * stage, const char * detail = nullptr);

// RAII: log RSS on construction and destruction of a scope.
class dist_rss_scope {
public:
    dist_rss_scope(const char * stage, const char * detail = nullptr);
    ~dist_rss_scope();

    dist_rss_scope(const dist_rss_scope &) = delete;
    dist_rss_scope & operator=(const dist_rss_scope &) = delete;

private:
    std::string stage_;
    std::string detail_;
    uint64_t    rss_enter_ = 0;
};
