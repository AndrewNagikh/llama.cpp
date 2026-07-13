#include "dist_process.h"

#include <cstdio>
#include <cstring>
#include <sstream>

#if defined(_WIN32)
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#else
#include <errno.h>
#include <fcntl.h>
#include <signal.h>
#include <sys/wait.h>
#include <unistd.h>
#endif

bool dist_set_env(const char * name, const char * value) {
    if (!name || !value) {
        return false;
    }
#if defined(_WIN32)
    return _putenv_s(name, value) == 0;
#else
    return setenv(name, value, 1) == 0;
#endif
}

std::string dist_home_dir() {
#if defined(_WIN32)
    if (const char * home = std::getenv("USERPROFILE")) {
        return home;
    }
#endif
    if (const char * home = std::getenv("HOME")) {
        return home;
    }
    return ".";
}

std::string dist_python_cmd() {
#if defined(_WIN32)
    return "python";
#else
    return "python3";
#endif
}

std::string dist_exe_suffix() {
#if defined(_WIN32)
    return ".exe";
#else
    return "";
#endif
}

std::string dist_join_path(const std::string & dir, const std::string & name) {
    if (dir.empty()) {
        return name;
    }
    const char last = dir.back();
    if (last == '/' || last == '\\') {
        return dir + name;
    }
#if defined(_WIN32)
    return dir + "\\" + name;
#else
    return dir + "/" + name;
#endif
}

void dist_sleep_ms(const int ms) {
    if (ms <= 0) {
        return;
    }
#if defined(_WIN32)
    Sleep(static_cast<DWORD>(ms));
#else
    usleep(static_cast<useconds_t>(ms) * 1000);
#endif
}

#if defined(_WIN32)

static std::string quote_win_arg(const std::string & arg) {
    if (arg.empty()) {
        return "\"\"";
    }
    bool needs_quotes = false;
    for (char c : arg) {
        if (c == ' ' || c == '\t' || c == '"') {
            needs_quotes = true;
            break;
        }
    }
    if (!needs_quotes) {
        return arg;
    }
    std::string out = "\"";
    for (char c : arg) {
        if (c == '"') {
            out += "\\\"";
        } else {
            out += c;
        }
    }
    out += '"';
    return out;
}

static std::string build_win_command_line(const std::vector<std::string> & argv) {
    std::ostringstream oss;
    for (size_t i = 0; i < argv.size(); ++i) {
        if (i > 0) {
            oss << ' ';
        }
        oss << quote_win_arg(argv[i]);
    }
    return oss.str();
}

static std::string dist_resolve_win_executable(const std::string & exe) {
    if (exe.find('\\') != std::string::npos || exe.find('/') != std::string::npos) {
        return exe;
    }
    const char * path_env = std::getenv("PATH");
    if (!path_env) {
        return exe;
    }
    std::string path = path_env;
    for (size_t start = 0; start < path.size();) {
        const size_t end = path.find(';', start);
        std::string dir  = path.substr(start, end == std::string::npos ? std::string::npos : end - start);
        if (!dir.empty()) {
            const std::string candidate = dist_join_path(dir, exe);
            const DWORD attr = GetFileAttributesA(candidate.c_str());
            if (attr != INVALID_FILE_ATTRIBUTES && !(attr & FILE_ATTRIBUTE_DIRECTORY)) {
                return candidate;
            }
        }
        if (end == std::string::npos) {
            break;
        }
        start = end + 1;
    }
    const std::string system32 = dist_join_path(
            std::getenv("WINDIR") ? std::getenv("WINDIR") : "C:\\Windows",
            "System32");
    const std::string system_curl = dist_join_path(system32, exe);
    const DWORD sys_attr = GetFileAttributesA(system_curl.c_str());
    if (sys_attr != INVALID_FILE_ATTRIBUTES && !(sys_attr & FILE_ATTRIBUTE_DIRECTORY)) {
        return system_curl;
    }
    return exe;
}

static std::vector<std::string> dist_resolve_win_argv(const std::vector<std::string> & argv) {
    if (argv.empty()) {
        return argv;
    }
    std::vector<std::string> resolved = argv;
    resolved[0] = dist_resolve_win_executable(argv[0]);
    return resolved;
}

bool dist_process_spawn(
        const std::vector<std::string> & argv,
        dist_child_process & out,
        std::string & err) {
    out = {};
    if (argv.empty()) {
        err = "empty argv";
        return false;
    }

    std::vector<std::string> win_argv = dist_resolve_win_argv(argv);
    std::string cmdline = build_win_command_line(win_argv);
    std::vector<char> cmdline_buf(cmdline.begin(), cmdline.end());
    cmdline_buf.push_back('\0');

    STARTUPINFOA si{};
    si.cb = sizeof(si);
    PROCESS_INFORMATION pi{};

    if (!CreateProcessA(
            win_argv[0].c_str(),
            cmdline_buf.data(),
            nullptr,
            nullptr,
            FALSE,
            CREATE_NO_WINDOW,
            nullptr,
            nullptr,
            &si,
            &pi)) {
        err = "CreateProcess failed: " + std::to_string(GetLastError());
        return false;
    }

    CloseHandle(pi.hThread);
    out.pid     = static_cast<uint64_t>(pi.dwProcessId);
    out.handle  = pi.hProcess;
    return true;
}

bool dist_process_kill(dist_child_process & proc) {
    if (proc.handle == nullptr) {
        return true;
    }
    TerminateProcess(static_cast<HANDLE>(proc.handle), 1);
    return true;
}

bool dist_process_reap(dist_child_process & proc) {
    if (proc.handle == nullptr) {
        proc.pid = 0;
        return true;
    }
    WaitForSingleObject(static_cast<HANDLE>(proc.handle), INFINITE);
    CloseHandle(static_cast<HANDLE>(proc.handle));
    proc.handle = nullptr;
    proc.pid    = 0;
    return true;
}

bool dist_process_is_running(dist_child_process & proc) {
    if (proc.handle == nullptr) {
        proc.pid = 0;
        return false;
    }
    DWORD code = 0;
    if (!GetExitCodeProcess(static_cast<HANDLE>(proc.handle), &code)) {
        return false;
    }
    if (code == STILL_ACTIVE) {
        return true;
    }
    CloseHandle(static_cast<HANDLE>(proc.handle));
    proc.handle = nullptr;
    proc.pid    = 0;
    return false;
}

int dist_process_run_capture_stdout(
        const std::vector<std::string> & argv,
        std::vector<uint8_t> & stdout_bytes,
        std::string & err) {
    stdout_bytes.clear();
    if (argv.empty()) {
        err = "empty argv";
        return -1;
    }

    SECURITY_ATTRIBUTES sa{};
    sa.nLength              = sizeof(sa);
    sa.bInheritHandle       = TRUE;
    sa.lpSecurityDescriptor = nullptr;

    HANDLE read_pipe  = nullptr;
    HANDLE write_pipe = nullptr;
    if (!CreatePipe(&read_pipe, &write_pipe, &sa, 0)) {
        err = "CreatePipe failed";
        return -1;
    }
    SetHandleInformation(read_pipe, HANDLE_FLAG_INHERIT, 0);

    std::vector<std::string> win_argv = dist_resolve_win_argv(argv);
    std::string cmdline = build_win_command_line(win_argv);
    std::vector<char> cmdline_buf(cmdline.begin(), cmdline.end());
    cmdline_buf.push_back('\0');

    STARTUPINFOA si{};
    si.cb         = sizeof(si);
    si.dwFlags    = STARTF_USESTDHANDLES;
    si.hStdOutput = write_pipe;
    si.hStdError  = GetStdHandle(STD_ERROR_HANDLE);
    si.hStdInput  = GetStdHandle(STD_INPUT_HANDLE);

    PROCESS_INFORMATION pi{};
    if (!CreateProcessA(
            win_argv[0].c_str(),
            cmdline_buf.data(),
            nullptr,
            nullptr,
            TRUE,
            CREATE_NO_WINDOW,
            nullptr,
            nullptr,
            &si,
            &pi)) {
        CloseHandle(read_pipe);
        CloseHandle(write_pipe);
        err = "CreateProcess failed: " + std::to_string(GetLastError());
        return -1;
    }

    CloseHandle(write_pipe);
    CloseHandle(pi.hThread);

    char buf[65536];
    DWORD nread = 0;
    while (ReadFile(read_pipe, buf, sizeof(buf), &nread, nullptr) && nread > 0) {
        stdout_bytes.insert(stdout_bytes.end(), buf, buf + nread);
    }
    CloseHandle(read_pipe);

    WaitForSingleObject(pi.hProcess, INFINITE);
    DWORD exit_code = 1;
    GetExitCodeProcess(pi.hProcess, &exit_code);
    CloseHandle(pi.hProcess);
    return static_cast<int>(exit_code);
}

#else

bool dist_process_spawn(
        const std::vector<std::string> & argv,
        dist_child_process & out,
        std::string & err) {
    out = {};
    if (argv.empty()) {
        err = "empty argv";
        return false;
    }

    // Build argv before fork(): node_agent runs multiple sync/download
    // worker threads, and any heap allocation in the child between fork()
    // and exec()/_exit() is undefined behavior in a multithreaded parent
    // (the child can inherit the malloc lock held by a thread that no
    // longer exists in it -- observed as "Heap corruption detected" on
    // macOS). cargs must already be fully materialized before fork().
    std::vector<char *> cargs;
    cargs.reserve(argv.size() + 1);
    for (const auto & a : argv) {
        cargs.push_back(const_cast<char *>(a.c_str()));
    }
    cargs.push_back(nullptr);

    const pid_t pid = fork();
    if (pid < 0) {
        err = "fork failed";
        return false;
    }
    if (pid == 0) {
        // execvp (not execv): argv[0] is often a bare command name (e.g.
        // "curl") that must be resolved via $PATH, not a literal path.
        execvp(argv[0].c_str(), cargs.data());
        _exit(127);
    }

    out.pid = static_cast<uint64_t>(pid);
    return true;
}

bool dist_process_kill(dist_child_process & proc) {
    if (proc.pid <= 0) {
        return true;
    }
    kill(static_cast<pid_t>(proc.pid), SIGTERM);
    return true;
}

bool dist_process_reap(dist_child_process & proc) {
    if (proc.pid <= 0) {
        return true;
    }
    waitpid(static_cast<pid_t>(proc.pid), nullptr, 0);
    proc.pid = 0;
    return true;
}

bool dist_process_is_running(dist_child_process & proc) {
    if (proc.pid <= 0) {
        return false;
    }
    int status = 0;
    const pid_t r = waitpid(static_cast<pid_t>(proc.pid), &status, WNOHANG);
    if (r == 0) {
        return true;
    }
    if (r == static_cast<pid_t>(proc.pid)) {
        proc.pid = 0;
        return false;
    }
    if (errno == ECHILD) {
        proc.pid = 0;
    }
    return false;
}

int dist_process_run_capture_stdout(
        const std::vector<std::string> & argv,
        std::vector<uint8_t> & stdout_bytes,
        std::string & err) {
    stdout_bytes.clear();
    if (argv.empty()) {
        err = "empty argv";
        return -1;
    }

    int pipefd[2];
    if (pipe(pipefd) != 0) {
        err = "pipe failed";
        return -1;
    }
    // Sync runs several concurrent download threads, each forking its own
    // curl child via this same function. Without FD_CLOEXEC, a fork() on
    // another thread between this pipe() and this thread's own fork() below
    // copies these fds into an unrelated child, which then holds the write
    // end open for its lifetime -- delaying or corrupting this thread's EOF
    // on read(). dup2() below creates STDOUT_FILENO fresh (no CLOEXEC), so
    // this still reaches the intended child correctly.
    fcntl(pipefd[0], F_SETFD, FD_CLOEXEC);
    fcntl(pipefd[1], F_SETFD, FD_CLOEXEC);

    // Build argv before fork() -- see dist_process_spawn for why (heap
    // allocation in the child of a multithreaded parent is unsafe).
    std::vector<char *> cargs;
    cargs.reserve(argv.size() + 1);
    for (const auto & a : argv) {
        cargs.push_back(const_cast<char *>(a.c_str()));
    }
    cargs.push_back(nullptr);

    const pid_t pid = fork();
    if (pid < 0) {
        close(pipefd[0]);
        close(pipefd[1]);
        err = "fork failed";
        return -1;
    }

    if (pid == 0) {
        close(pipefd[0]);
        dup2(pipefd[1], STDOUT_FILENO);
        close(pipefd[1]);

        // execvp (not execv): argv[0] is often a bare command name (e.g.
        // "curl") that must be resolved via $PATH, not a literal path.
        execvp(argv[0].c_str(), cargs.data());
        _exit(127);
    }

    close(pipefd[1]);
    char buf[65536];
    ssize_t nread = 0;
    while ((nread = read(pipefd[0], buf, sizeof(buf))) > 0) {
        stdout_bytes.insert(stdout_bytes.end(), buf, buf + nread);
    }
    close(pipefd[0]);

    int status = 0;
    if (waitpid(pid, &status, 0) < 0) {
        err = "waitpid failed";
        return -1;
    }
    if (!WIFEXITED(status)) {
        return -1;
    }
    return WEXITSTATUS(status);
}

#endif
