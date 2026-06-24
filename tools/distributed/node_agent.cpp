#include "dist_common.h"
#include "model_catalog.h"
#include "node_benchmark.h"

#include "httplib.h"
#include "nlohmann/json.hpp"
#include "llama.h"

#include "ggml-backend.h"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <map>
#include <mutex>
#include <set>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#if !defined(_WIN32)
#include <signal.h>
#include <sys/wait.h>
#include <unistd.h>
#endif

using json = nlohmann::json;

static std::string g_model_path;
static std::string g_node_id;
static int32_t g_n_layer = 0;
static int32_t g_n_embd  = 0;
static BenchmarkResult g_benchmark{};
static pid_t g_worker_pid = 0;
static model_store g_model_store;
static std::mutex g_model_store_mu;
static std::set<std::string> g_active_downloads;
static std::map<std::string, std::string> g_download_errors;
static std::string g_models_state_path;

static uint64_t now_ms() {
    const auto now = std::chrono::system_clock::now();
    const auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(now.time_since_epoch());
    return static_cast<uint64_t>(ms.count());
}

static std::string shell_quote(const std::string & value) {
    std::string quoted = "'";
    for (char c : value) {
        if (c == '\'') {
            quoted += "'\\''";
        } else {
            quoted += c;
        }
    }
    quoted += "'";
    return quoted;
}

static uint64_t file_size_or_zero(const std::string & path) {
    std::error_code ec;
    const auto size = std::filesystem::file_size(path, ec);
    return ec ? 0 : static_cast<uint64_t>(size);
}

static void usage(const char * prog) {
    fprintf(stderr,
            "usage: %s --model PATH --listen HOST:PORT --orchestrator URL "
            "[--node-id ID] [--advertise-host IP] [--rebenchmark]\n"
            "example: %s --model llama.gguf --listen 0.0.0.0:9001 --orchestrator http://10.0.0.1:9000 --advertise-host 10.0.0.2\n",
            prog, prog);
}

static bool parse_args(int argc, char ** argv, std::string & listen, std::string & orchestrator,
        std::string & node_id, std::string & advertise_host, bool & rebenchmark) {
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "--model") == 0 && i + 1 < argc) {
            g_model_path = argv[++i];
        } else if (strcmp(argv[i], "--listen") == 0 && i + 1 < argc) {
            listen = argv[++i];
        } else if (strcmp(argv[i], "--orchestrator") == 0 && i + 1 < argc) {
            orchestrator = argv[++i];
        } else if (strcmp(argv[i], "--node-id") == 0 && i + 1 < argc) {
            node_id = argv[++i];
        } else if (strcmp(argv[i], "--advertise-host") == 0 && i + 1 < argc) {
            advertise_host = argv[++i];
        } else if (strcmp(argv[i], "--rebenchmark") == 0) {
            rebenchmark = true;
        } else {
            return false;
        }
    }
    return !g_model_path.empty() && !listen.empty() && !orchestrator.empty();
}

static std::string exe_dir(const char * argv0) {
    std::string p(argv0);
    const auto pos = p.find_last_of("/\\");
    if (pos != std::string::npos) {
        return p.substr(0, pos + 1);
    }
    return "./";
}

static void start_model_download(
        std::string model_id,
        std::string repo,
        std::string file,
        std::string output_path,
        std::string downloader_script) {
    std::thread([model_id = std::move(model_id),
                 repo = std::move(repo),
                 file = std::move(file),
                 output_path = std::move(output_path),
                 downloader_script = std::move(downloader_script)]() {
        const std::string progress_path = output_path + ".progress.json";
        const std::string command =
                "python3 " + shell_quote(downloader_script) +
                " --repo " + shell_quote(repo) +
                " --file " + shell_quote(file) +
                " --output " + shell_quote(output_path) +
                " --progress-file " + shell_quote(progress_path);

        const int rc = std::system(command.c_str());
        const bool ok = (rc == 0) && std::filesystem::exists(output_path) && file_size_or_zero(output_path) > 0;

        std::lock_guard<std::mutex> lock(g_model_store_mu);
        installed_model model;
        model.model_id = model_id;
        model.local_path = output_path;
        model.size_bytes = file_size_or_zero(output_path);
        model.ready = ok;
        model.installed_ms = ok ? now_ms() : 0;
        g_model_store.add_model(model);

        g_active_downloads.erase(model_id);
        if (ok) {
            g_download_errors.erase(model_id);
        } else {
            g_download_errors[model_id] = "download failed";
        }
        g_model_store.save_state(g_models_state_path);
    }).detach();
}

static bool load_model_metadata() {
    ggml_backend_load_all();
    llama_model * model = llama_model_load_from_file(g_model_path.c_str(), llama_model_default_params());
    if (!model) {
        return false;
    }
    g_n_layer = llama_model_n_layer(model);
    g_n_embd  = llama_model_n_embd(model);
    llama_model_free(model);
    return true;
}

static dist_configure_req parse_configure(const json & body) {
    dist_configure_req req{};
    const std::string role = body.value("role", "");
    if (role == "entry") {
        req.role = DIST_ROLE_ENTRY;
    } else if (role == "middle") {
        req.role = DIST_ROLE_MIDDLE;
    } else if (role == "final") {
        req.role = DIST_ROLE_FINAL;
    }
    req.layer_start = body.value("layer_start", 0);
    req.layer_end   = body.value("layer_end", 0);
    req.session_id  = body.value("session_id", "");
    req.ctrl_port   = body.value("ctrl_port", 0);
    req.peer_port   = body.value("peer_port", 0);
    req.next_host   = body.value("next_host", "127.0.0.1");
    req.next_port   = body.value("next_port", 0);
    req.peer_bind   = body.value("peer_bind", "0.0.0.0");
    return req;
}

static bool register_with_orchestrator(
        const std::string & orchestrator,
        const std::string & node_id,
        const std::string & host,
        int http_port) {
    httplib::Client cli(orchestrator.c_str());
    cli.set_connection_timeout(5, 0);
    cli.set_read_timeout(10, 0);

    int64_t total_mb = 0;
    int64_t free_mb  = 0;
    dist_probe_memory(total_mb, free_mb);
    const auto caps = dist_probe_capabilities();

    const double score = g_benchmark.score;

    json body = {
        { "node_id", node_id },
        { "host", host },
        { "port", http_port },
        { "n_layer", g_n_layer },
        { "n_embd", g_n_embd },
        { "score", score },
        { "memory_total_mb", total_mb },
        { "memory_free_mb", free_mb },
        { "hardware", {
            { "cpu_threads", caps.cpu_threads },
            { "ram_gb", (int) ((total_mb + 512) / 1024) },
            { "gpu_name", caps.gpu_name },
            { "gpu_vram_gb", caps.gpu_memory_mb / 1024 },
        }},
        { "capabilities", {
            { "gpu_backend", caps.gpu_backend },
            { "gpu_memory_mb", caps.gpu_memory_mb },
            { "cpu_threads", caps.cpu_threads },
            { "supported_arch", caps.supported_arch },
        }},
        { "performance", {
            { "score", score },
            { "decode_tps", g_benchmark.decode_tps },
            { "prefill_tps", g_benchmark.prefill_tps },
            { "load_ms", g_benchmark.load_ms },
        }},
    };

    const auto res = cli.Post("/register", body.dump(), "application/json");
    if (!res || res->status != 200) {
        if (!res) {
            fprintf(stderr, "node_agent: register failed — cannot reach orchestrator at %s\n",
                    orchestrator.c_str());
        } else {
            fprintf(stderr, "node_agent: register failed HTTP %d: %s\n",
                    res->status, res->body.c_str());
        }
        return false;
    }

    fprintf(stderr, "node_agent: registered as %s at %s:%d score=%.1f\n",
            node_id.c_str(), host.c_str(), http_port, score);
    return true;
}

#if !defined(_WIN32)

static void stop_worker() {
    if (g_worker_pid > 0) {
        kill(g_worker_pid, SIGTERM);
        waitpid(g_worker_pid, nullptr, 0);
        g_worker_pid = 0;
    }
}

static bool start_worker(const std::string & agent_bin, const dist_configure_req & cfg, std::string & err) {
    stop_worker();

    const std::string dir = agent_bin;
    std::string bin;
    std::vector<std::string> args;

    if (cfg.role == DIST_ROLE_ENTRY) {
        bin = dir + "split_gen3_a";
        args = {
            bin, g_model_path,
            "--ctrl-port", std::to_string(cfg.ctrl_port),
            "--b-host", cfg.next_host,
            "--b-port", std::to_string(cfg.next_port),
            "--layer-end", std::to_string(cfg.layer_end),
            "--bind", cfg.peer_bind,
        };
    } else if (cfg.role == DIST_ROLE_MIDDLE) {
        bin = dir + "split_gen3_b";
        args = {
            bin, g_model_path,
            "--ab-port", std::to_string(cfg.peer_port),
            "--bc-host", cfg.next_host,
            "--bc-port", std::to_string(cfg.next_port),
            "--layer-start", std::to_string(cfg.layer_start),
            "--layer-end", std::to_string(cfg.layer_end),
            "--bind", cfg.peer_bind,
        };
    } else if (cfg.role == DIST_ROLE_FINAL) {
        bin = dir + "split_gen3_c";
        args = {
            bin, g_model_path,
            "--bc-port", std::to_string(cfg.peer_port),
            "--layer-start", std::to_string(cfg.layer_start),
            "--bind", cfg.peer_bind,
        };
    } else {
        err = "invalid role";
        return false;
    }

    const pid_t pid = fork();
    if (pid < 0) {
        err = "fork failed";
        return false;
    }

    if (pid == 0) {
        std::vector<char *> cargs;
        cargs.reserve(args.size() + 1);
        for (auto & a : args) {
            cargs.push_back(a.data());
        }
        cargs.push_back(nullptr);
        execv(cargs[0], cargs.data());
        _exit(127);
    }

    g_worker_pid = pid;
    usleep(300000);
    return true;
}

#endif

int main(int argc, char ** argv) {
    std::string listen;
    std::string orchestrator;
    std::string advertise_host;
    bool rebenchmark = false;

    if (!parse_args(argc, argv, listen, orchestrator, g_node_id, advertise_host, rebenchmark)) {
        usage(argv[0]);
        return 1;
    }

    std::string bind_host;
    int http_port = 0;
    if (!dist_parse_host_port(listen, bind_host, http_port)) {
        fprintf(stderr, "node_agent: invalid --listen %s\n", listen.c_str());
        return 1;
    }

    if (g_node_id.empty()) {
        g_node_id = bind_host + ":" + std::to_string(http_port);
    }

    fprintf(stderr, "node_agent: running benchmark on %s\n", g_model_path.c_str());
    g_benchmark = dist_get_or_run_benchmark(g_model_path, rebenchmark);
    if (g_benchmark.score <= 0.0f) {
        fprintf(stderr, "node_agent: benchmark failed\n");
        return 1;
    }

    g_n_layer = g_benchmark.n_layer;
    g_n_embd  = g_benchmark.n_embd;
    if (g_n_layer <= 0 || g_n_embd <= 0) {
        if (!load_model_metadata()) {
            fprintf(stderr, "node_agent: failed to read model metadata from %s\n", g_model_path.c_str());
            return 1;
        }
    }

    // Initialize model store
    g_models_state_path = g_model_store.get_models_dir() + "/models.json";
    if (!g_model_store.load_state(g_models_state_path)) {
        fprintf(stderr, "node_agent: warning - failed to load model state from %s\n", g_models_state_path.c_str());
    }

    std::string register_host = !advertise_host.empty() ? advertise_host : bind_host;
    if (register_host == "0.0.0.0" || register_host == "*") {
        fprintf(stderr, "node_agent: set --advertise-host to a reachable IP (not 0.0.0.0)\n");
        return 1;
    }

    if (!register_with_orchestrator(orchestrator, g_node_id, register_host, http_port)) {
        return 1;
    }

    const std::string agent_bin = exe_dir(argv[0]);

    httplib::Server svr;

    svr.Get("/health", [](const httplib::Request &, httplib::Response & res) {
        res.set_content(json({ { "status", "ok" }, { "node_id", g_node_id } }).dump(), "application/json");
    });

    svr.Get("/capabilities", [](const httplib::Request &, httplib::Response & res) {
        int64_t total_mb = 0;
        int64_t free_mb  = 0;
        dist_probe_memory(total_mb, free_mb);
        const auto caps = dist_probe_capabilities();

        res.set_content(json({
            { "node_id", g_node_id },
            { "hardware", {
                { "cpu_threads", caps.cpu_threads },
                { "ram_gb", (int) ((total_mb + 512) / 1024) },
                { "gpu_name", caps.gpu_name },
                { "gpu_vram_gb", caps.gpu_memory_mb / 1024 },
            }},
            { "performance", {
                { "score", g_benchmark.score },
                { "decode_tps", g_benchmark.decode_tps },
                { "prefill_tps", g_benchmark.prefill_tps },
                { "load_ms", g_benchmark.load_ms },
            }},
        }).dump(), "application/json");
    });

    svr.Get("/status", [](const httplib::Request &, httplib::Response & res) {
        res.set_content(json({
            { "node_id", g_node_id },
            { "worker_pid", (int) g_worker_pid },
        }).dump(), "application/json");
    });

    svr.Post("/configure", [agent_bin](const httplib::Request & req, httplib::Response & res) {
#if defined(_WIN32)
        res.status = 501;
        res.set_content(R"({"error":"configure unsupported on Windows"})", "application/json");
        return;
#else
        json body;
        try {
            body = json::parse(req.body);
        } catch (...) {
            res.status = 400;
            res.set_content(R"({"error":"invalid json"})", "application/json");
            return;
        }

        const dist_configure_req cfg = parse_configure(body);
        std::string err;
        if (!start_worker(agent_bin, cfg, err)) {
            res.status = 500;
            res.set_content(json({ { "ok", false }, { "error", err } }).dump(), "application/json");
            return;
        }

        res.set_content(json({
            { "ok", true },
            { "role", dist_role_name(cfg.role) },
            { "worker_pid", (int) g_worker_pid },
        }).dump(), "application/json");
#endif
    });

    svr.Post("/shutdown", [](const httplib::Request &, httplib::Response & res) {
#if !defined(_WIN32)
        stop_worker();
#endif
        res.set_content(R"({"ok":true})", "application/json");
    });

    // Model store API
    
    // GET /models/local - List locally installed models
    svr.Get("/models/local", [](const httplib::Request &, httplib::Response & res) {
        json models_json = json::array();

        std::lock_guard<std::mutex> lock(g_model_store_mu);
        auto local_models = g_model_store.get_local_models();
        for (const auto & model : local_models) {
            std::string status = model.ready ? "ready" : "unknown";
            const auto active_it = g_active_downloads.find(model.model_id);
            if (active_it != g_active_downloads.end()) {
                status = "downloading";
            }
            const auto error_it = g_download_errors.find(model.model_id);
            if (error_it != g_download_errors.end()) {
                status = "error";
            }

            json model_json = {
                { "model_id", model.model_id },
                { "local_path", model.local_path },
                { "size_bytes", model.size_bytes },
                { "ready", model.ready },
                { "status", status },
                { "installed_ms", model.installed_ms }
            };
            if (error_it != g_download_errors.end()) {
                model_json["error"] = error_it->second;
            }
            models_json.push_back(model_json);
        }
        
        res.set_content(models_json.dump(), "application/json");
    });
    
    // POST /models/install - Install a model locally
    svr.Post("/models/install", [agent_bin](const httplib::Request & req, httplib::Response & res) {
        json body;
        try {
            body = json::parse(req.body);
        } catch (...) {
            res.status = 400;
            res.set_content(R"({"error":"invalid json"})", "application/json");
            return;
        }
        
        const std::string model_id = body.value("model", "");
        const std::string repo = body.value("repo", "");
        const std::string file = body.value("file", "");
        
        if (model_id.empty() || repo.empty() || file.empty()) {
            res.status = 400;
            res.set_content(R"({"error":"model, repo, and file required"})", "application/json");
            return;
        }
        
        const std::string expected_path = g_model_store.get_model_path(model_id);
        const std::string downloader_script = agent_bin + "../../tools/distributed/download_model.py";

        {
            std::lock_guard<std::mutex> lock(g_model_store_mu);
            const auto * existing = g_model_store.find_model(model_id);
            if (existing && existing->ready && std::filesystem::exists(existing->local_path)) {
                res.set_content(json({
                    { "model_id", model_id },
                    { "status", "already_installed" },
                    { "local_path", existing->local_path }
                }).dump(), "application/json");
                return;
            }

            if (g_active_downloads.find(model_id) != g_active_downloads.end()) {
                res.set_content(json({
                    { "model_id", model_id },
                    { "status", "download_in_progress" },
                    { "local_path", expected_path }
                }).dump(), "application/json");
                return;
            }

            installed_model new_model;
            new_model.model_id = model_id;
            new_model.local_path = expected_path;
            new_model.ready = false;
            new_model.size_bytes = file_size_or_zero(expected_path);
            new_model.installed_ms = 0;

            g_model_store.add_model(new_model);
            g_download_errors.erase(model_id);
            g_active_downloads.insert(model_id);
            g_model_store.save_state(g_models_state_path);
        }

        start_model_download(model_id, repo, file, expected_path, downloader_script);

        res.set_content(json({
            { "model_id", model_id },
            { "status", "download_started" },
            { "local_path", expected_path }
        }).dump(), "application/json");
    });

    fprintf(stderr, "node_agent: listening on %s:%d model=%s score=%.1f\n",
            bind_host.c_str(), http_port, g_model_path.c_str(), g_benchmark.score);

    if (!svr.listen(bind_host.c_str(), http_port)) {
        fprintf(stderr, "node_agent: failed to bind %s:%d\n", bind_host.c_str(), http_port);
#if !defined(_WIN32)
        stop_worker();
#endif
        return 1;
    }

#if !defined(_WIN32)
    stop_worker();
#endif
    return 0;
}
