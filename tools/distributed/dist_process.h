#pragma once

#include <cstdint>
#include <string>
#include <vector>

// Cross-platform child process helpers (Unix fork/exec, Windows CreateProcess).

struct dist_child_process {
    uint64_t pid = 0;
#if defined(_WIN32)
    void * handle = nullptr; // HANDLE
#endif
};

bool dist_set_env(const char * name, const char * value);
std::string dist_home_dir();
std::string dist_python_cmd();
std::string dist_exe_suffix();
std::string dist_join_path(const std::string & dir, const std::string & name);

void dist_sleep_ms(int ms);

// log_path: if non-empty, the child's stdout/stderr are redirected there
// (append mode). On POSIX the child already inherits the parent's
// stdout/stderr when log_path is empty; on Windows CreateProcess neither
// inherits nor opens a console for the child, so without a log_path its
// output -- including crash diagnostics -- is silently lost.
bool dist_process_spawn(
        const std::vector<std::string> & argv,
        dist_child_process & out,
        std::string & err,
        const std::string & log_path = std::string());

bool dist_process_kill(dist_child_process & proc);
bool dist_process_reap(dist_child_process & proc);
bool dist_process_is_running(dist_child_process & proc);

// Run command, capture stdout. Returns exit code (-1 on spawn error).
int dist_process_run_capture_stdout(
        const std::vector<std::string> & argv,
        std::vector<uint8_t> & stdout_bytes,
        std::string & err);
