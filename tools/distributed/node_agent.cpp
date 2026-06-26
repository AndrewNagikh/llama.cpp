#include "dist_common.h"
#include "model_catalog.h"
#include "node_benchmark.h"
#include "node_agent/layer_store/layer_store.h"
#include "node_agent/layer_store/layer_gguf_assembler.h"
#include "node_agent/synchronization/synchronization_engine.h"
#include "orchestrator/install_planner/install_planner.h"
#include "orchestrator/coverage/coverage.h"
#include "orchestrator/manifest_builder/manifest_builder.h"

#include "httplib.h"
#include "nlohmann/json.hpp"
#include "llama.h"

#include "ggml-backend.h"
#include "transport/split_tcp_wire.h"

#include <chrono>
#include <cctype>
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
static int g_pipeline_ctrl_port = 0;
static int g_pipeline_layer_end = 0;
static model_store g_model_store;
static std::mutex g_model_store_mu;
static std::set<std::string> g_active_downloads;
static std::map<std::string, std::string> g_download_errors;
static std::string g_models_state_path;
static std::string g_models_dir;
static synchronization_engine g_sync_engine;
static std::mutex g_layer_store_mu;

static layer_store get_layer_store(const std::string & model_id) {
    std::lock_guard<std::mutex> lock(g_layer_store_mu);
    layer_store store(g_model_store.get_models_dir(), model_id);
    return store;
}

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
            "usage: %s --listen HOST:PORT --orchestrator URL "
            "[--model PATH] [--node-id ID] [--advertise-host IP] [--models-dir DIR] [--rebenchmark]\n"
            "  Layer-first mode: omit --model; workers load GGUF assembled from synced layers.\n"
            "example: %s --listen 0.0.0.0:9001 --orchestrator http://10.0.0.1:9000 --advertise-host 10.0.0.2\n",
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
        } else if (strcmp(argv[i], "--models-dir") == 0 && i + 1 < argc) {
            g_models_dir = argv[++i];
        } else if (strcmp(argv[i], "--rebenchmark") == 0) {
            rebenchmark = true;
        } else {
            return false;
        }
    }
    if (g_models_dir.empty()) {
        if (const char * env = std::getenv("MODELS_DIR")) {
            g_models_dir = env;
        } else if (const char * home = std::getenv("HOME")) {
            g_models_dir = std::string(home) + "/.distributed-llm/models";
        }
    }
    return !listen.empty() && !orchestrator.empty();
}

static std::string exe_dir(const char * argv0) {
    std::string p(argv0);
    const auto pos = p.find_last_of("/\\");
    if (pos != std::string::npos) {
        return p.substr(0, pos + 1);
    }
    return "./";
}

// Locate an already-present GGUF that satisfies the requested catalog file,
// so install can seed locally (symlink) instead of downloading over the network.
static std::string find_local_source(const std::string & file) {
    if (file.empty()) {
        return "";
    }
    // The node already runs against a model file; reuse it when names match.
    if (!g_model_path.empty()) {
        std::error_code ec;
        if (std::filesystem::exists(g_model_path, ec)) {
            const std::string base = std::filesystem::path(g_model_path).filename().string();
            if (base == file) {
                return g_model_path;
            }
        }
    }
    return "";
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
    req.model_id    = body.value("model_id", "");
    req.ctrl_port   = body.value("ctrl_port", 0);
    req.peer_port   = body.value("peer_port", 0);
    req.next_host   = body.value("next_host", "127.0.0.1");
    req.next_port   = body.value("next_port", 0);
    req.peer_bind   = body.value("peer_bind", "0.0.0.0");
    req.next_is_final = body.value("next_is_final", false);
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

    const double score = g_benchmark.score;

    dist_node_memory memory{};
    dist_node_cpu cpu{};
    const dist_node_system sys = dist_probe_node_system();
    dist_probe_node_memory(memory);
    dist_probe_node_cpu(cpu);
    const auto caps = dist_probe_capabilities();

    const auto ram_gb  = static_cast<int>(memory.total_ram_bytes / (1024 * 1024 * 1024));
    const auto vram_gb = static_cast<int>(memory.total_vram_bytes / (1024 * 1024 * 1024));

    json body = {
        { "node_id", node_id },
        { "host", host },
        { "port", http_port },
        { "n_layer", g_n_layer },
        { "n_embd", g_n_embd },
        { "score", score },
        { "memory_total_mb", static_cast<int64_t>(memory.total_ram_bytes / (1024 * 1024)) },
        { "memory_free_mb", static_cast<int64_t>(memory.free_ram_bytes / (1024 * 1024)) },
        { "memory", {
            { "total_ram", memory.total_ram_bytes },
            { "free_ram", memory.free_ram_bytes },
            { "total_vram", memory.total_vram_bytes },
            { "free_vram", memory.free_vram_bytes },
            { "has_gpu", memory.has_gpu },
        }},
        { "hardware", {
            { "backend", caps.gpu_backend },
            { "gpu_name", caps.gpu_name },
            { "cpu_name", cpu.cpu_name },
            { "logical_cores", cpu.logical_cores },
            { "physical_cores", cpu.physical_cores },
            { "cpu_threads", caps.cpu_threads },
            { "ram_gb", ram_gb },
            { "gpu_vram_gb", vram_gb },
        }},
        { "capabilities", {
            { "gpu_backend", caps.gpu_backend },
            { "gpu_memory_mb", caps.gpu_memory_mb },
            { "cpu_threads", caps.cpu_threads },
            { "supported_arch", caps.supported_arch },
        }},
        { "system", {
            { "os", sys.os },
            { "arch", sys.arch },
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
    g_pipeline_ctrl_port = 0;
    g_pipeline_layer_end = 0;
}

static bool pipeline_gen3_send_recv(
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

static bool run_local_pipeline_generate(
        const std::vector<int32_t> & prompt_tokens,
        int max_new,
        int layer_end,
        std::vector<int32_t> & out_tokens,
        std::string & err) {
    if (g_pipeline_ctrl_port <= 0) {
        err = "pipeline not configured";
        return false;
    }

    int ctrl_fd = split_tcp_connect_retry("127.0.0.1", g_pipeline_ctrl_port, 300, 100);
    if (ctrl_fd < 0) {
        err = "failed to connect to local pipeline ctrl port";
        return false;
    }

    split_gen3_a_resp resp{};
    if (!pipeline_gen3_send_recv(ctrl_fd, SPLIT_GEN_CMD_RESET, 0, 0, layer_end, nullptr, resp)) {
        err = "reset failed";
        close(ctrl_fd);
        return false;
    }

    if (!pipeline_gen3_send_recv(ctrl_fd, SPLIT_GEN_CMD_PREFILL, (int32_t) prompt_tokens.size(), 0,
            layer_end, prompt_tokens.data(), resp)) {
        err = "prefill failed";
        close(ctrl_fd);
        return false;
    }

    if (resp.token_id < 0) {
        err = "prefill returned error from pipeline";
        close(ctrl_fd);
        return false;
    }

    out_tokens.push_back(resp.token_id);
    int32_t cur = resp.token_id;

    const int n_prompt = (int) prompt_tokens.size();
    for (int step = 1; step < max_new; ++step) {
        const int32_t pos = n_prompt + step - 1;
        if (!pipeline_gen3_send_recv(ctrl_fd, SPLIT_GEN_CMD_DECODE, 1, pos, layer_end, &cur, resp)) {
            err = "decode failed at step " + std::to_string(step);
            close(ctrl_fd);
            return false;
        }
        if (resp.token_id < 0) {
            err = "pipeline error at step " + std::to_string(step);
            close(ctrl_fd);
            return false;
        }
        cur = resp.token_id;
        out_tokens.push_back(cur);
    }

    pipeline_gen3_send_recv(ctrl_fd, SPLIT_GEN_CMD_SHUTDOWN, 0, 0, layer_end, nullptr, resp);
    close(ctrl_fd);
    return true;
}

static bool start_worker(
        const std::string & agent_bin,
        const dist_configure_req & cfg,
        const std::string & model_path,
        std::string & err) {
    stop_worker();

    if (model_path.empty()) {
        err = "no model path for worker";
        return false;
    }

    const std::string dir = agent_bin;
    std::string bin;
    std::vector<std::string> args;

    if (cfg.role == DIST_ROLE_ENTRY) {
        bin = dir + "split_gen3_a";
        args = {
            bin, model_path,
            "--ctrl-port", std::to_string(cfg.ctrl_port),
            "--b-host", cfg.next_host,
            "--b-port", std::to_string(cfg.next_port),
            "--layer-end", std::to_string(cfg.layer_end),
            "--bind", cfg.peer_bind,
        };
        if (cfg.next_is_final) {
            args.push_back("--next-final");
        }
    } else if (cfg.role == DIST_ROLE_MIDDLE) {
        bin = dir + "split_gen3_b";
        args = {
            bin, model_path,
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
            bin, model_path,
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

static std::string materialize_worker_gguf(
        const dist_configure_req & cfg,
        std::string & err) {
    if (cfg.model_id.empty()) {
        err = "model_id required for layer-first configure";
        return {};
    }

    layer_store store = get_layer_store(cfg.model_id);
    const auto manifest = store.load_manifest();
    if (!manifest.has_value()) {
        err = "manifest not found in layer store; run install/sync first";
        return {};
    }

    const bool include_embedding = (cfg.role == DIST_ROLE_ENTRY);
    const bool include_output    = (cfg.role == DIST_ROLE_FINAL);
    const std::string role_name  = dist_role_name(cfg.role);
    const std::string out_path =
            (store.model_root() / ("worker_" + role_name + ".gguf")).string();

    if (!layer_store_materialize_gguf(
                store,
                *manifest,
                out_path,
                cfg.layer_start,
                cfg.layer_end,
                include_embedding,
                include_output)) {
        err = "failed to assemble GGUF from layer store for role " + role_name;
        return {};
    }

    fprintf(stderr,
            "node_agent: materialized %s from layers [%d,%d) embedding=%d output=%d\n",
            out_path.c_str(),
            cfg.layer_start,
            cfg.layer_end,
            include_embedding ? 1 : 0,
            include_output ? 1 : 0);
    return out_path;
}

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

    if (!g_model_path.empty()) {
        fprintf(stderr, "node_agent: running benchmark on %s\n", g_model_path.c_str());
    } else {
        fprintf(stderr, "node_agent: no local model — hardware-only benchmark\n");
    }
    g_benchmark = dist_get_or_run_benchmark_optional(g_model_path, rebenchmark);
    if (g_benchmark.score <= 0.0f) {
        fprintf(stderr, "node_agent: benchmark failed\n");
        return 1;
    }

    g_n_layer = g_benchmark.n_layer;
    g_n_embd  = g_benchmark.n_embd;
    if (g_n_layer <= 0 || g_n_embd <= 0) {
        if (!g_model_path.empty() && !load_model_metadata()) {
            fprintf(stderr, "node_agent: failed to read model metadata from %s\n", g_model_path.c_str());
            return 1;
        }
    }

    // Initialize model store
    if (!g_models_dir.empty()) {
        g_model_store.set_models_dir(g_models_dir);
    }
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

    // Heartbeat: periodically re-register so the orchestrator recovers this node
    // after an orchestrator restart (registration is upserted by node_id).
    std::thread([orchestrator, register_host, http_port]() {
        for (;;) {
            std::this_thread::sleep_for(std::chrono::seconds(5));
            register_with_orchestrator(orchestrator, g_node_id, register_host, http_port);
        }
    }).detach();

    const std::string agent_bin = exe_dir(argv[0]);

    httplib::Server svr;

    svr.Get("/health", [](const httplib::Request &, httplib::Response & res) {
        res.set_content(json({ { "status", "ok" }, { "node_id", g_node_id } }).dump(), "application/json");
    });

    svr.Get("/capabilities", [](const httplib::Request &, httplib::Response & res) {
        dist_node_memory memory{};
        dist_node_cpu cpu{};
        const dist_node_system sys = dist_probe_node_system();
        dist_probe_node_memory(memory);
        dist_probe_node_cpu(cpu);
        const auto caps = dist_probe_capabilities();

        res.set_content(json({
            { "node_id", g_node_id },
            { "memory", {
                { "total_ram", memory.total_ram_bytes },
                { "free_ram", memory.free_ram_bytes },
                { "total_vram", memory.total_vram_bytes },
                { "free_vram", memory.free_vram_bytes },
                { "has_gpu", memory.has_gpu },
            }},
            { "hardware", {
                { "backend", caps.gpu_backend },
                { "gpu_name", caps.gpu_name },
                { "cpu_name", cpu.cpu_name },
                { "logical_cores", cpu.logical_cores },
                { "physical_cores", cpu.physical_cores },
            }},
            { "system", {
                { "os", sys.os },
                { "arch", sys.arch },
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
        std::string worker_model = g_model_path;
        std::string err;

        if (!cfg.model_id.empty()) {
            const std::string materialized = materialize_worker_gguf(cfg, err);
            if (materialized.empty()) {
                res.status = 500;
                res.set_content(json({ { "ok", false }, { "error", err } }).dump(), "application/json");
                return;
            }
            worker_model = materialized;
        } else if (worker_model.empty()) {
            res.status = 400;
            res.set_content(json({
                { "ok", false },
                { "error", "model_id or local --model required" },
            }).dump(), "application/json");
            return;
        }

        if (!start_worker(agent_bin, cfg, worker_model, err)) {
            res.status = 500;
            res.set_content(json({ { "ok", false }, { "error", err } }).dump(), "application/json");
            return;
        }

        if (cfg.role == DIST_ROLE_ENTRY) {
            g_pipeline_ctrl_port = cfg.ctrl_port;
            g_pipeline_layer_end   = cfg.layer_end;
        }

        res.set_content(json({
            { "ok", true },
            { "role", dist_role_name(cfg.role) },
            { "worker_pid", (int) g_worker_pid },
        }).dump(), "application/json");
#endif
    });

    svr.Post("/pipeline/generate", [](const httplib::Request & req, httplib::Response & res) {
#if defined(_WIN32)
        res.status = 501;
        res.set_content(R"({"ok":false,"error":"pipeline generate unsupported on Windows"})", "application/json");
        return;
#else
        json body;
        try {
            body = json::parse(req.body);
        } catch (...) {
            res.status = 400;
            res.set_content(R"({"ok":false,"error":"invalid json"})", "application/json");
            return;
        }

        const int max_tokens = body.value("max_tokens", 16);
        const int layer_end  = body.value("layer_end", g_pipeline_layer_end);

        std::vector<int32_t> prompt_tokens;
        if (body.contains("prompt_tokens") && body["prompt_tokens"].is_array()) {
            for (const auto & t : body["prompt_tokens"]) {
                prompt_tokens.push_back(t.get<int32_t>());
            }
        }

        if (prompt_tokens.empty()) {
            res.status = 400;
            res.set_content(R"({"ok":false,"error":"prompt_tokens required"})", "application/json");
            return;
        }
        if (layer_end <= 0) {
            res.status = 503;
            res.set_content(R"({"ok":false,"error":"pipeline not configured"})", "application/json");
            return;
        }

        std::vector<int32_t> out_tokens;
        std::string err;
        if (!run_local_pipeline_generate(prompt_tokens, max_tokens, layer_end, out_tokens, err)) {
            res.status = 500;
            res.set_content(json({ { "ok", false }, { "error", err }, { "tokens", json::array() } }).dump(),
                    "application/json");
            return;
        }

        json tokens_json = json::array();
        for (const int32_t t : out_tokens) {
            tokens_json.push_back(t);
        }
        res.set_content(json({ { "ok", true }, { "tokens", tokens_json } }).dump(), "application/json");
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

    // GET /installed-layers - Report per-layer state from Layer Store (Task 9.7).
    svr.Get("/installed-layers", [](const httplib::Request & req, httplib::Response & res) {
        const std::string model_id = req.get_param_value("model");
        json layers_json = json::array();

        if (!model_id.empty()) {
            const dist_node_capabilities caps = dist_probe_capabilities();
            const std::string device = dist_normalize_device(caps.gpu_backend, caps.has_gpu);
            const layer_store store = get_layer_store(model_id);

            for (const layer_blob & blob : store.list_layers()) {
                std::string state = "READY";
                if (!store.verify_layer(blob.layer_index, blob.checksum)) {
                    state = "CORRUPTED";
                }

                layers_json.push_back({
                    { "layer", blob.layer_index },
                    { "node", g_node_id },
                    { "node_id", g_node_id },
                    { "device", device },
                    { "size_bytes", blob.size_bytes },
                    { "checksum", blob.checksum },
                    { "state", state },
                });
            }
        }

        res.set_content(json({
            { "model", model_id },
            { "layers", layers_json },
        }).dump(), "application/json");
    });

    // POST /models/{model_id}/install/execute - Run Synchronization Engine locally.
    svr.Post(R"(/models/([^/]+)/install/execute)", [](const httplib::Request & req, httplib::Response & res) {
        const std::string model_id = req.matches[1];

        json body;
        try {
            body = json::parse(req.body);
        } catch (...) {
            res.status = 400;
            res.set_content(R"({"error":"invalid json"})", "application/json");
            return;
        }

        if (!body.contains("operations") || !body["operations"].is_array()) {
            res.status = 400;
            res.set_content(R"({"error":"operations array required"})", "application/json");
            return;
        }

        std::vector<install_operation> operations;
        for (const auto & item : body["operations"]) {
            operations.push_back(install_operation::from_json(item));
        }

        const model_manifest * manifest_ptr = nullptr;
        model_manifest manifest;
        if (body.contains("manifest") && body["manifest"].is_object()) {
            manifest = model_manifest::from_json(body["manifest"]);
            manifest_ptr = &manifest;
        }

        layer_store store = get_layer_store(model_id);
        const std::string job_id = g_sync_engine.start_job(
                model_id, operations, store, manifest_ptr);

        res.set_content(json({
            { "job_id", job_id },
            { "model_id", model_id },
            { "status", "started" },
            { "operation_count", static_cast<int>(operations.size()) },
        }).dump(), "application/json");
    });

    // GET /jobs/{job_id} - Synchronization job progress.
    svr.Get(R"(/jobs/(.+))", [](const httplib::Request & req, httplib::Response & res) {
        const std::string job_id = req.matches[1];
        const auto job = g_sync_engine.get_job(job_id);
        if (!job.has_value()) {
            res.status = 404;
            res.set_content(R"({"error":"job not found"})", "application/json");
            return;
        }
        res.set_content(job->to_json().dump(), "application/json");
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

            // Fast path: seed from an already-present GGUF (symlink) instead of
            // downloading. Keeps installs offline/fast when the file is local.
            const std::string local_src = find_local_source(file);
            if (!local_src.empty()) {
                std::error_code ec;
                std::filesystem::create_directories(
                        std::filesystem::path(expected_path).parent_path(), ec);
                std::filesystem::remove(expected_path, ec);
                std::filesystem::create_symlink(local_src, expected_path, ec);
                if (ec) {
                    // Fall back to a copy if symlinks are unsupported.
                    ec.clear();
                    std::filesystem::copy_file(local_src, expected_path,
                            std::filesystem::copy_options::overwrite_existing, ec);
                }
                if (!ec && file_size_or_zero(expected_path) > 0) {
                    installed_model seeded;
                    seeded.model_id = model_id;
                    seeded.local_path = expected_path;
                    seeded.size_bytes = file_size_or_zero(expected_path);
                    seeded.ready = true;
                    seeded.installed_ms = now_ms();
                    g_model_store.add_model(seeded);
                    g_download_errors.erase(model_id);
                    g_model_store.save_state(g_models_state_path);
                    res.set_content(json({
                        { "model_id", model_id },
                        { "status", "already_installed" },
                        { "local_path", expected_path }
                    }).dump(), "application/json");
                    return;
                }
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

    const char * model_label = g_model_path.empty() ? "(layer-store)" : g_model_path.c_str();
    fprintf(stderr, "node_agent: listening on %s:%d model=%s score=%.1f\n",
            bind_host.c_str(), http_port, model_label, g_benchmark.score);

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
