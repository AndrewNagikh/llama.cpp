#pragma once

// Shared helpers for the end-to-end cluster validation suite (Task 7.5).
// Header-only: each test binary is a single translation unit that includes this.

#include "dist_common.h"
#include "split_gen3_common.h"
#include "split_gen_common.h"
#include "split_tcp_wire.h"

#include "httplib.h"
#include "nlohmann/json.hpp"
#include "llama.h"

#include "ggml-backend.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <string>
#include <vector>

#if !defined(_WIN32)
#include <arpa/inet.h>
#include <netinet/in.h>
#include <signal.h>
#include <sys/socket.h>
#include <sys/wait.h>
#include <unistd.h>
#endif

namespace e2e {

using json = nlohmann::json;

inline std::string log_dir() {
    return "logs/e2e";
}

inline std::string exe_dir(const char * argv0) {
    std::string p(argv0);
    const auto pos = p.find_last_of("/\\");
    if (pos != std::string::npos) {
        return p.substr(0, pos + 1);
    }
    return "./";
}

// Locate a GGUF model file: explicit override, then $MODEL, then default path.
inline std::string find_gguf(const std::string & override_path = "") {
    std::vector<std::string> candidates;
    if (!override_path.empty()) {
        candidates.push_back(override_path);
    }
    if (const char * env = std::getenv("MODEL")) {
        candidates.push_back(env);
    }
    if (const char * home = std::getenv("HOME")) {
        candidates.push_back(std::string(home) + "/models/llama-3.2-1b-instruct-q4_k_m.gguf");
    }
    for (const auto & c : candidates) {
        std::error_code ec;
        if (!c.empty() && std::filesystem::exists(c, ec)) {
            return c;
        }
    }
    return "";
}

// ----------------------------------------------------------------------------
// HTTP helpers
// ----------------------------------------------------------------------------

inline bool wait_http_ok(const std::string & url, int retries = 100) {
    for (int i = 0; i < retries; ++i) {
        httplib::Client cli(url.c_str());
        cli.set_connection_timeout(1, 0);
        const auto res = cli.Get("/health");
        if (res && res->status == 200) {
            return true;
        }
        usleep(100000);
    }
    return false;
}

inline bool http_get(const std::string & base, const std::string & path, json & out, int & status,
        int read_timeout = 15) {
    httplib::Client cli(base.c_str());
    cli.set_connection_timeout(3, 0);
    cli.set_read_timeout(read_timeout, 0);
    const auto res = cli.Get(path.c_str());
    status = res ? res->status : 0;
    if (!res) {
        return false;
    }
    try {
        out = json::parse(res->body);
    } catch (...) {
        return false;
    }
    return true;
}

inline bool http_post(const std::string & base, const std::string & path, const json & body,
        json & out, int & status, int read_timeout = 120) {
    httplib::Client cli(base.c_str());
    cli.set_connection_timeout(5, 0);
    cli.set_read_timeout(read_timeout, 0);
    const auto res = cli.Post(path.c_str(), body.dump(), "application/json");
    status = res ? res->status : 0;
    if (!res) {
        return false;
    }
    try {
        out = json::parse(res->body);
    } catch (...) {
        return false;
    }
    return true;
}

inline std::string model_basename(const std::string & gguf_path) {
    return std::filesystem::path(gguf_path).filename().string();
}

inline bool register_model(const std::string & orch, const std::string & model_id,
        const std::string & gguf_path, std::string & err,
        const std::string & repository = "") {
    json out;
    int status = 0;
    const std::string repo = repository.empty()
            ? ("hugging-quants/" + model_id + "-GGUF")
            : repository;
    json body = {
        { "model_id", model_id },
        { "display_name", model_id },
        { "source", "huggingface" },
        { "repository", repo },
        { "filename", model_basename(gguf_path) },
        { "revision", "main" },
    };
    if (!http_post(orch, "/models/register", body, out, status, 15) || status != 200) {
        err = "register request failed status=" + std::to_string(status);
        return false;
    }
    if (out.value("status", "") != "DISCOVERED") {
        err = "unexpected status after register: " + out.value("status", "");
        return false;
    }
    return true;
}

inline bool discover_model(const std::string & orch, const std::string & model_id,
        std::string & err, int expected_files = 1) {
    json out;
    int status = 0;
    if (!http_post(orch, "/models/" + model_id + "/discover", json({}), out, status, 120) || status != 200) {
        err = "discover request failed status=" + std::to_string(status) + " body=" + out.dump();
        return false;
    }
    if (out.value("status", "") != "ok") {
        err = "unexpected discover status: " + out.dump();
        return false;
    }
    if (out.value("files", 0) < expected_files) {
        err = "discover returned too few files: " + out.dump();
        return false;
    }

    json model;
    int gs = 0;
    if (!http_get(orch, "/models/" + model_id, model, gs, 15) || gs != 200) {
        err = "get model after discovery failed status=" + std::to_string(gs);
        return false;
    }
    if (model.value("status", "") != "MANIFEST_PENDING") {
        err = "unexpected registry status after discovery: " + model.value("status", "");
        return false;
    }
    return true;
}

// ----------------------------------------------------------------------------
// Process management (Unix only)
// ----------------------------------------------------------------------------

#if !defined(_WIN32)

// Spawn a child process, redirecting its stdout/stderr to logfile.
inline pid_t spawn_logged(const std::vector<std::string> & args, const std::string & logfile) {
    const pid_t pid = fork();
    if (pid != 0) {
        return pid;
    }

    // child
    if (!logfile.empty()) {
        std::filesystem::create_directories(std::filesystem::path(logfile).parent_path());
        FILE * f = freopen(logfile.c_str(), "a", stdout);
        (void) f;
        f = freopen(logfile.c_str(), "a", stderr);
        (void) f;
    }

    std::vector<char *> cargs;
    cargs.reserve(args.size() + 1);
    for (const auto & a : args) {
        cargs.push_back(const_cast<char *>(a.c_str()));
    }
    cargs.push_back(nullptr);

    execv(cargs[0], cargs.data());
    _exit(127);
}

inline bool pid_alive(pid_t pid) {
    if (pid <= 0) {
        return false;
    }
    return kill(pid, 0) == 0;
}

inline void kill_proc(pid_t pid) {
    if (pid > 0) {
        kill(pid, SIGTERM);
    }
}

inline void kill_wait(pid_t pid) {
    if (pid > 0) {
        kill(pid, SIGTERM);
        waitpid(pid, nullptr, 0);
    }
}

// Return true if a TCP port is free to bind on 127.0.0.1.
inline bool port_free(int port) {
    const int fd = socket(AF_INET, SOCK_STREAM, 0);
    if (fd < 0) {
        return false;
    }
    int yes = 1;
    setsockopt(fd, SOL_SOCKET, SO_REUSEADDR, &yes, sizeof(yes));

    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    addr.sin_port = htons((uint16_t) port);

    const bool ok = bind(fd, (sockaddr *) &addr, sizeof(addr)) == 0;
    close(fd);
    return ok;
}

#endif // !_WIN32

// ----------------------------------------------------------------------------
// Cluster helpers
// ----------------------------------------------------------------------------

inline int count_online_nodes(const json & nodes_resp) {
    if (!nodes_resp.contains("nodes") || !nodes_resp["nodes"].is_array()) {
        return 0;
    }
    int n = 0;
    for (const auto & node : nodes_resp["nodes"]) {
        if (node.value("online", false)) {
            ++n;
        }
    }
    return n;
}

// Wait until the orchestrator reports at least `expected` online nodes.
inline bool wait_nodes_registered(const std::string & orch, int expected, int retries = 150) {
    for (int i = 0; i < retries; ++i) {
        json out;
        int status = 0;
        if (http_get(orch, "/nodes", out, status, 5) && status == 200) {
            if (count_online_nodes(out) >= expected) {
                return true;
            }
        }
        usleep(200000);
    }
    return false;
}

// Check GET /models for a model that is ready on at least `expected_nodes`.
inline bool models_ready(const std::string & orch, const std::string & model, int expected_nodes) {
    json out;
    int status = 0;
    if (!http_get(orch, "/models/installed", out, status, 15) || status != 200 || !out.is_array()) {
        return false;
    }
    for (const auto & m : out) {
        if (m.value("id", "") == model) {
            return m.value("ready_nodes", 0) >= expected_nodes;
        }
    }
    return false;
}

// Start a cluster-wide install and wait until the model is ready on all nodes.
inline bool install_and_wait_ready(const std::string & orch, const std::string & model,
        int expected_nodes, std::string & err, int timeout_s = 1800) {
    json install_out;
    int status = 0;
    json body = { { "model", model } };
    if (!http_post(orch, "/models/install", body, install_out, status, 30) || status != 200) {
        err = "install request failed status=" + std::to_string(status);
        return false;
    }

    const std::string job_id = install_out.value("job_id", "");
    if (job_id.empty()) {
        err = "install returned no job_id";
        return false;
    }

    const int max_polls = timeout_s; // poll every ~1s
    for (int i = 0; i < max_polls; ++i) {
        json job;
        int js = 0;
        if (http_get(orch, "/models/install/" + job_id, job, js, 10) && js == 200) {
            const std::string st = job.value("status", "unknown");
            if (st == "ready") {
                // Confirm aggregated view also reports readiness on all nodes.
                if (models_ready(orch, model, expected_nodes)) {
                    return true;
                }
            } else if (st == "error") {
                err = "install job error: " + job.value("error", "");
                return false;
            }
        }
        // Even if job already "ready" but models view lags, keep polling models.
        if (models_ready(orch, model, expected_nodes)) {
            return true;
        }
        sleep(1);
    }

    err = "install did not reach ready within timeout";
    return false;
}

// Validate that a session layout covers [0, n_layer) contiguously across all
// expected nodes with strictly increasing, non-overlapping ranges.
inline bool validate_layout(const json & layout, int n_layer, int expected_nodes, std::string & err) {
    if (!layout.is_array() || layout.empty()) {
        err = "layout is empty";
        return false;
    }

    struct seg { int start; int end; };
    std::vector<seg> segs;
    for (const auto & item : layout) {
        if (!item.contains("layer_start") || !item.contains("layer_end")) {
            err = "layout entry missing layer bounds";
            return false;
        }
        const int s = item.value("layer_start", -1);
        const int e = item.value("layer_end", -1);
        if (s < 0 || e < 0 || s >= e) {
            err = "invalid range layer_start=" + std::to_string(s) + " layer_end=" + std::to_string(e);
            return false;
        }
        segs.push_back({ s, e });
    }

    if ((int) segs.size() != expected_nodes) {
        err = "layout has " + std::to_string(segs.size()) + " stages, expected " +
              std::to_string(expected_nodes);
        return false;
    }

    std::sort(segs.begin(), segs.end(), [](const seg & a, const seg & b) { return a.start < b.start; });

    if (segs.front().start != 0) {
        err = "coverage does not start at 0";
        return false;
    }
    if (segs.back().end != n_layer) {
        err = "coverage does not end at n_layer=" + std::to_string(n_layer) +
              " (got " + std::to_string(segs.back().end) + ")";
        return false;
    }
    for (size_t i = 1; i < segs.size(); ++i) {
        if (segs[i].start != segs[i - 1].end) {
            err = "gap or overlap between [" + std::to_string(segs[i - 1].start) + "," +
                  std::to_string(segs[i - 1].end) + ") and [" + std::to_string(segs[i].start) +
                  "," + std::to_string(segs[i].end) + ")";
            return false;
        }
    }
    return true;
}

inline int model_n_layer(const char * model_path) {
    ggml_backend_load_all();
    llama_model * model = llama_model_load_from_file(model_path, llama_model_default_params());
    if (!model) {
        return -1;
    }
    const int n = llama_model_n_layer(model);
    llama_model_free(model);
    return n;
}

// Index of the first end-of-generation token (e.g. <|eot_id|>/<|end_of_text|>)
// in `toks`, or -1 if none. Used to bound exact token comparison to the
// "real" generated content before the model decides to stop.
inline int first_eog_index(const char * model_path, const std::vector<llama_token> & toks) {
    ggml_backend_load_all();
    llama_model * model = llama_model_load_from_file(model_path, llama_model_default_params());
    if (!model) {
        return -1;
    }
    const llama_vocab * vocab = llama_model_get_vocab(model);
    int idx = -1;
    for (size_t i = 0; i < toks.size(); ++i) {
        if (llama_vocab_is_eog(vocab, toks[i])) {
            idx = (int) i;
            break;
        }
    }
    llama_model_free(model);
    return idx;
}

// ----------------------------------------------------------------------------
// Reference generation: in-process 3-way split baseline (greedy)
// ----------------------------------------------------------------------------

#if !defined(_WIN32)

inline bool gen3_send_recv(
        int ctrl_fd,
        split_gen_cmd cmd,
        int32_t n_tokens,
        int32_t pos_start,
        int32_t layer_end,
        const int32_t * tokens,
        split_gen3_a_resp & resp) {
    if (!split_gen_send_req(ctrl_fd, cmd, n_tokens, pos_start, layer_end, 0, tokens)) {
        return false;
    }
    return split_gen3_recv_a_resp(ctrl_fd, resp, nullptr);
}

// Runs split_gen3 a/b/c workers locally and returns the greedy-decoded tokens.
inline std::vector<llama_token> run_split3_baseline(
        const std::string & dir, const char * model_path, const std::string & prompt, int max_new) {
    const split_gen3_layout layout = SPLIT_GEN3_LAYOUTS[0];
    const int base = 24000 + (getpid() % 500);
    const int bc_port   = base;
    const int ab_port   = base + 1;
    const int ctrl_port = base + 2;

    pid_t pid_a = 0, pid_b = 0, pid_c = 0;

    pid_c = fork();
    if (pid_c == 0) {
        const std::string bc = std::to_string(bc_port);
        const std::string ls = std::to_string(layout.layer_c_start);
        execl((dir + "split_gen3_c").c_str(), "split_gen3_c", model_path,
                "--bc-port", bc.c_str(), "--layer-start", ls.c_str(), (char *) nullptr);
        _exit(127);
    }

    usleep(500000);

    pid_b = fork();
    if (pid_b == 0) {
        const std::string ab = std::to_string(ab_port);
        const std::string bc = std::to_string(bc_port);
        const std::string ls = std::to_string(layout.layer_b_start);
        const std::string le = std::to_string(layout.layer_b_end);
        execl((dir + "split_gen3_b").c_str(), "split_gen3_b", model_path,
                "--ab-port", ab.c_str(), "--bc-port", bc.c_str(),
                "--layer-start", ls.c_str(), "--layer-end", le.c_str(), (char *) nullptr);
        _exit(127);
    }

    usleep(500000);

    pid_a = fork();
    if (pid_a == 0) {
        const std::string ctrl = std::to_string(ctrl_port);
        const std::string ab   = std::to_string(ab_port);
        const std::string le   = std::to_string(layout.layer_a_end);
        execl((dir + "split_gen3_a").c_str(), "split_gen3_a", model_path,
                "--ctrl-port", ctrl.c_str(), "--b-port", ab.c_str(),
                "--layer-end", le.c_str(), (char *) nullptr);
        _exit(127);
    }

    int ctrl_fd = -1;
    for (int retry = 0; retry < 300; ++retry) {
        ctrl_fd = split_tcp_connect("127.0.0.1", ctrl_port);
        if (ctrl_fd >= 0) {
            break;
        }
        usleep(100000);
    }

    std::vector<llama_token> out;
    auto cleanup = [&]() {
        if (ctrl_fd >= 0) {
            close(ctrl_fd);
        }
        kill_wait(pid_a);
        kill_wait(pid_b);
        kill_wait(pid_c);
    };

    if (ctrl_fd < 0) {
        cleanup();
        return out;
    }

    ggml_backend_load_all();
    llama_model * model = llama_model_load_from_file(model_path, llama_model_default_params());
    if (!model) {
        cleanup();
        return out;
    }

    const auto ptokens = split_gen_tokenize(llama_model_get_vocab(model), prompt);
    llama_model_free(model);

    split_gen3_a_resp resp{};
    gen3_send_recv(ctrl_fd, SPLIT_GEN_CMD_RESET, 0, 0, layout.layer_a_end, nullptr, resp);

    std::vector<int32_t> ptoks(ptokens.size());
    for (size_t i = 0; i < ptokens.size(); ++i) {
        ptoks[i] = (int32_t) ptokens[i];
    }

    if (!gen3_send_recv(ctrl_fd, SPLIT_GEN_CMD_PREFILL, (int32_t) ptokens.size(), 0,
            layout.layer_a_end, ptoks.data(), resp)) {
        cleanup();
        return out;
    }

    out.push_back((llama_token) resp.token_id);
    llama_token cur = (llama_token) resp.token_id;

    for (int step = 1; step < max_new; ++step) {
        const int32_t tok = (int32_t) cur;
        const int32_t pos = (int32_t) ptokens.size() + step - 1;
        if (!gen3_send_recv(ctrl_fd, SPLIT_GEN_CMD_DECODE, 1, pos, layout.layer_a_end, &tok, resp)) {
            break;
        }
        cur = (llama_token) resp.token_id;
        out.push_back(cur);
    }

    gen3_send_recv(ctrl_fd, SPLIT_GEN_CMD_SHUTDOWN, 0, 0, layout.layer_a_end, nullptr, resp);
    cleanup();
    return out;
}

#endif // !_WIN32

// ----------------------------------------------------------------------------
// Summary reporting
// ----------------------------------------------------------------------------

struct stage_result {
    std::string name;
    bool pass = false;
    std::string detail;
};

inline void write_summary(const std::string & path, const std::vector<stage_result> & stages, bool overall) {
    json out;
    out["overall"] = overall ? "PASS" : "FAIL";
    json arr = json::array();
    for (const auto & s : stages) {
        arr.push_back({
            { "stage", s.name },
            { "result", s.pass ? "PASS" : "FAIL" },
            { "detail", s.detail },
        });
    }
    out["stages"] = arr;

    std::error_code ec;
    std::filesystem::create_directories(std::filesystem::path(path).parent_path(), ec);
    FILE * f = fopen(path.c_str(), "w");
    if (f) {
        const std::string dump = out.dump(2);
        fwrite(dump.data(), 1, dump.size(), f);
        fclose(f);
    }
}

inline void print_report(const std::vector<stage_result> & stages, bool overall) {
    printf("\n==================== E2E REPORT ====================\n");
    for (const auto & s : stages) {
        printf("  %-28s %s%s%s\n", s.name.c_str(), s.pass ? "PASS" : "FAIL",
                s.detail.empty() ? "" : "  -- ", s.detail.c_str());
    }
    printf("---------------------------------------------------\n");
    printf("  OVERALL RESULT: %s\n", overall ? "PASS" : "FAIL");
    printf("===================================================\n");
}

} // namespace e2e
