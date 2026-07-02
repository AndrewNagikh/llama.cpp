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

bool dist_process_spawn(
        const std::vector<std::string> & argv,
        dist_child_process & out,
        std::string & err) {
    out = {};
    if (argv.empty()) {
        err = "empty argv";
        return false;
    }

    std::string cmdline = build_win_command_line(argv);
    std::vector<char> cmdline_buf(cmdline.begin(), cmdline.end());
    cmdline_buf.push_back('\0');

    STARTUPINFOA si{};
    si.cb = sizeof(si);
    PROCESS_INFORMATION pi{};

    if (!CreateProcessA(
            argv[0].c_str(),
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

    std::string cmdline = build_win_command_line(argv);
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
            argv[0].c_str(),
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
        execv(argv[0].c_str(), cargs.data());
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

        std::vector<char *> cargs;
        cargs.reserve(argv.size() + 1);
        for (const auto & a : argv) {
            cargs.push_back(const_cast<char *>(a.c_str()));
        }
        cargs.push_back(nullptr);
        execv(argv[0].c_str(), cargs.data());
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
