#include "dist_common.h"
#include "dist_process.h"
#include "model_catalog.h"
#include "node_benchmark.h"
#include "node_agent/layer_store/layer_store.h"
#include "node_agent/layer_store/worker_builder.h"
#include "node_agent/layer_store/layer_gguf_assembler.h"
#include "architecture/semantic_runtime_descriptor.h"
#include "architecture/worker_requirement.h"
#include "runtime/runtime_role.h"
#include "runtime/runtime_worker_bind.h"
#include "verification/worker_verify.h"
#include "node_agent/synchronization/synchronization_engine.h"
#include "orchestrator/install_planner/install_planner.h"
#include "orchestrator/coverage/coverage.h"
#include "orchestrator/manifest_builder/manifest_builder.h"

#include "httplib.h"
#include "nlohmann/json.hpp"
#include "llama.h"

#include "ggml-backend.h"
#include "transport/split_tcp_wire.h"
#include "workers/split_gen_common.h"
#include "runtime_debug/runtime_debug.h"
#include "runtime_debug/trace_recorder.h"

#include <chrono>
#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <map>
#include <mutex>
#include <set>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

using json = nlohmann::json;

static std::string g_model_path;
static std::string g_node_id;
static int32_t g_n_layer = 0;
static int32_t g_n_embd  = 0;
static BenchmarkResult g_benchmark{};
static dist_child_process g_entry_worker{};
static dist_child_process g_middle_worker{};
static dist_child_process g_final_worker{};
static int g_pipeline_ctrl_port = 0;
static int g_pipeline_layer_end = 0;

struct node_runtime_stats {
    int configure_count        = 0;
    int materialization_count  = 0;
    int worker_spawn_count     = 0;
    int pipeline_generate_count = 0;
    int kv_cache_reset_count   = 0;
    int context_create_count   = 0;
    int runtime_load_count     = 0;
    int materialization_generation = 0;
    int tokenizer_init_count   = 0;
};

static node_runtime_stats g_runtime_stats;
static std::string g_entry_worker_gguf;
static std::string g_tokenizer_service_gguf;
static std::string g_embedding_service_gguf;
static std::string g_output_service_gguf;
static bool g_sampler_service_ready = false;
static llama_model * g_entry_tokenizer = nullptr;
static llama_model * g_tokenizer_service_model = nullptr;

static void free_tokenizer_service() {
    if (g_tokenizer_service_model) {
        llama_model_free(g_tokenizer_service_model);
        g_tokenizer_service_model = nullptr;
    }
}

static void free_entry_tokenizer() {
    if (g_entry_tokenizer) {
        llama_model_free(g_entry_tokenizer);
        g_entry_tokenizer = nullptr;
    }
}

static const llama_vocab * tokenizer_service_vocab() {
    if (!g_tokenizer_service_model && !g_tokenizer_service_gguf.empty()) {
        ggml_backend_load_all();
        llama_model_params params = llama_model_default_params();
        params.vocab_only = true;
        g_tokenizer_service_model = llama_model_load_from_file(g_tokenizer_service_gguf.c_str(), params);
        if (g_tokenizer_service_model) {
            g_runtime_stats.tokenizer_init_count = 1;
        }
    }
    return g_tokenizer_service_model ? llama_model_get_vocab(g_tokenizer_service_model) : nullptr;
}

static const llama_vocab * entry_node_vocab() {
    if (!g_entry_tokenizer && !g_entry_worker_gguf.empty()) {
        ggml_backend_load_all();
        llama_model_params params = llama_model_default_params();
        params.vocab_only = true;
        g_entry_tokenizer = llama_model_load_from_file(g_entry_worker_gguf.c_str(), params);
        if (g_entry_tokenizer) {
            g_runtime_stats.tokenizer_init_count = 1;
        }
    }
    return g_entry_tokenizer ? llama_model_get_vocab(g_entry_tokenizer) : nullptr;
}

static json node_runtime_stats_json() {
    return {
        { "configure_count", g_runtime_stats.configure_count },
        { "materialization_count", g_runtime_stats.materialization_count },
        { "worker_spawn_count", g_runtime_stats.worker_spawn_count },
        { "pipeline_generate_count", g_runtime_stats.pipeline_generate_count },
        { "kv_cache_reset_count", g_runtime_stats.kv_cache_reset_count },
        { "context_create_count", g_runtime_stats.context_create_count },
        { "runtime_load_count", g_runtime_stats.runtime_load_count },
        { "materialization_generation", g_runtime_stats.materialization_generation },
        { "tokenizer_init_count", g_runtime_stats.tokenizer_init_count },
        { "materialized", g_runtime_stats.materialization_count > 0 },
        { "runtime_loaded", g_runtime_stats.runtime_load_count > 0 },
    };
}
static model_store g_model_store;
static std::mutex g_model_store_mu;
static std::set<std::string> g_active_downloads;
static std::map<std::string, std::string> g_download_errors;
static std::string g_models_state_path;
static std::string g_models_dir;
static bool g_verify_materialization = false;
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
            "[--model PATH] [--node-id ID] [--advertise-host IP] [--models-dir DIR] [--verify-materialization] [--rebenchmark] [--distributed-debug]\n"
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
        } else if (strcmp(argv[i], "--verify-materialization") == 0) {
            g_verify_materialization = true;
        } else if (strcmp(argv[i], "--rebenchmark") == 0) {
            rebenchmark = true;
        } else if (strcmp(argv[i], "--distributed-debug") == 0) {
            dist_set_env("LLAMA_DISTRIBUTED_DEBUG", "1");
        } else {
            return false;
        }
    }
    if (g_models_dir.empty()) {
        if (const char * env = std::getenv("MODELS_DIR")) {
            g_models_dir = env;
        } else {
            g_models_dir = dist_home_dir() + "/.distributed-llm/models";
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
                dist_python_cmd() + " " + shell_quote(downloader_script) +
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
    req.source_url    = body.value("source_url", "");
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

static dist_child_process * worker_proc_slot(const dist_node_role role) {
    switch (role) {
        case DIST_ROLE_ENTRY:  return &g_entry_worker;
        case DIST_ROLE_MIDDLE: return &g_middle_worker;
        case DIST_ROLE_FINAL:  return &g_final_worker;
        default:               return nullptr;
    }
}

static void stop_worker_for_role(const dist_node_role role) {
    dist_child_process * slot = worker_proc_slot(role);
    if (!slot || slot->pid == 0) {
        return;
    }
    dist_process_kill(*slot);
    dist_process_reap(*slot);
    *slot = {};
    if (role == DIST_ROLE_ENTRY) {
        g_pipeline_ctrl_port = 0;
        g_pipeline_layer_end = 0;
    }
}

static void stop_all_workers() {
    stop_worker_for_role(DIST_ROLE_ENTRY);
    stop_worker_for_role(DIST_ROLE_MIDDLE);
    stop_worker_for_role(DIST_ROLE_FINAL);
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
        std::string & err,
        json * timing_out = nullptr) {
    if (g_pipeline_ctrl_port <= 0) {
        err = "pipeline not configured";
        return false;
    }

    dist_debug_load_config();
    dist_debug_set_worker("pipeline", g_node_id);
    trace_recorder * dbg = dist_debug_recorder();
    int32_t debug_step   = 0;

    g_runtime_stats.pipeline_generate_count++;
    g_runtime_stats.context_create_count++;

    const auto t_total0 = std::chrono::steady_clock::now();
    int ctrl_fd = split_tcp_connect_retry("127.0.0.1", g_pipeline_ctrl_port, 300, 100);
    if (ctrl_fd < 0) {
        err = "failed to connect to local pipeline ctrl port";
        return false;
    }

    split_gen3_a_resp resp{};
    if (dbg) {
        dbg->emit_step_begin(debug_step, "reset", -1, 0, 0);
    }
    if (!pipeline_gen3_send_recv(ctrl_fd, SPLIT_GEN_CMD_RESET, 0, 0, layer_end, nullptr, resp)) {
        err = "reset failed";
        split_tcp_close(ctrl_fd);
        return false;
    }
    g_runtime_stats.kv_cache_reset_count++;

    const auto t_prefill0 = std::chrono::steady_clock::now();
    if (!pipeline_gen3_send_recv(ctrl_fd, SPLIT_GEN_CMD_PREFILL, (int32_t) prompt_tokens.size(), 0,
            layer_end, prompt_tokens.data(), resp)) {
        err = "prefill failed";
        split_tcp_close(ctrl_fd);
        return false;
    }

    const auto t_first_token = std::chrono::steady_clock::now();

    if (dbg) {
        dbg->emit_step_begin(debug_step, "prefill", -1, 0, 0);
    }

    if (resp.token_id < 0) {
        err = "prefill returned error from pipeline";
        split_tcp_close(ctrl_fd);
        return false;
    }

    out_tokens.push_back(resp.token_id);
    if (dbg) {
        dbg->emit_token_selected(debug_step, "prefill", resp.token_id,
                (int32_t) prompt_tokens.size() - 1, false);
    }
    debug_step++;
    int32_t cur = resp.token_id;

    const int n_prompt = (int) prompt_tokens.size();
    for (int step = 1; step < max_new; ++step) {
        const int32_t pos = n_prompt + step - 1;
        if (dbg) {
            dbg->emit_step_begin(debug_step, "decode", cur, pos, 0);
        }
        if (!pipeline_gen3_send_recv(ctrl_fd, SPLIT_GEN_CMD_DECODE, 1, pos, layer_end, &cur, resp)) {
            err = "decode failed at step " + std::to_string(step);
            split_tcp_close(ctrl_fd);
            return false;
        }
        if (resp.token_id < 0) {
            err = "pipeline error at step " + std::to_string(step);
            split_tcp_close(ctrl_fd);
            return false;
        }
        cur = resp.token_id;
        out_tokens.push_back(cur);
        if (dbg) {
            dbg->emit_token_selected(debug_step, "decode", cur, pos, false);
        }
        debug_step++;
    }

    split_tcp_close(ctrl_fd);

    const auto t_total1 = std::chrono::steady_clock::now();
    if (timing_out) {
        const double prefill_ms = std::chrono::duration<double, std::milli>(t_first_token - t_prefill0).count();
        const double total_ms = std::chrono::duration<double, std::milli>(t_total1 - t_total0).count();
        const double decode_ms = std::max(0.0, total_ms - prefill_ms);
        *timing_out = {
            { "prefill_ms", prefill_ms },
            { "ttft_ms", prefill_ms },
            { "decode_ms", decode_ms },
            { "total_ms", total_ms },
            { "generated_tokens", (int) out_tokens.size() },
        };
    }
    return true;
}

static bool start_worker(
        const std::string & agent_bin,
        const dist_configure_req & cfg,
        const std::string & model_path,
        std::string & err) {
    stop_worker_for_role(cfg.role);

    if (model_path.empty()) {
        err = "no model path for worker";
        return false;
    }

    const std::string suffix = dist_exe_suffix();
    const std::string dir    = agent_bin;
    std::string bin;
    std::vector<std::string> args;

    if (cfg.role == DIST_ROLE_ENTRY) {
        bin = dist_join_path(dir, "split_gen3_a" + suffix);
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
        bin = dist_join_path(dir, "split_gen3_b" + suffix);
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
        bin = dist_join_path(dir, "split_gen3_c" + suffix);
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

    if (dist_debug_enabled()) {
        dist_set_env("LLAMA_DISTRIBUTED_DEBUG", "1");
        if (!cfg.session_id.empty()) {
            dist_set_env("LLAMA_DIST_SESSION_ID", cfg.session_id.c_str());
        }
        dist_set_env("LLAMA_DIST_WORKER_ID", dist_role_name(cfg.role).c_str());
        dist_set_env("LLAMA_DIST_NODE_ID", g_node_id.c_str());
        if (!g_models_dir.empty()) {
            const std::string trace_dir = g_models_dir + "/traces";
            dist_set_env("LLAMA_DIST_TRACE_DIR", trace_dir.c_str());
        }
    }

    dist_child_process child{};
    if (!dist_process_spawn(args, child, err)) {
        return false;
    }

    if (dist_child_process * slot = worker_proc_slot(cfg.role)) {
        *slot = child;
    }
    g_runtime_stats.worker_spawn_count++;
    dist_sleep_ms(300);
    return true;
}

static worker_role dist_role_to_worker_role(const dist_node_role role) {
    switch (role) {
        case DIST_ROLE_ENTRY:  return worker_role::entry;
        case DIST_ROLE_MIDDLE: return worker_role::middle;
        case DIST_ROLE_FINAL:  return worker_role::final;
        default:               return worker_role::entry;
    }
}

static bool tokenizer_shell_loadable(const std::string & path) {
    if (path.empty()) {
        return false;
    }
    ggml_backend_load_all();
    llama_model_params params = llama_model_default_params();
    params.vocab_only = true;
    llama_model * model = llama_model_load_from_file(path.c_str(), params);
    if (!model) {
        return false;
    }
    llama_model_free(model);
    return true;
}

static std::string configure_materialize_worker_gguf(
        const dist_configure_req & cfg,
        const json & body,
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

    if (!store.metadata_bytes().has_value() && !cfg.source_url.empty()) {
        layer_store_cache_metadata(store, *manifest, cfg.source_url);
    }

    worker_role role = dist_role_to_worker_role(cfg.role);
    if (body.contains("runtime_role")) {
        const runtime_role rt = runtime_role_from_string(body.value("runtime_role", ""));
        if (rt == runtime_role::pipeline_stage && cfg.role != DIST_ROLE_UNCONFIGURED) {
            role = dist_role_to_worker_role(cfg.role);
        } else if (rt != runtime_role::unassigned) {
            role = worker_role_from_runtime_role(rt);
        }
    }

    if (role == worker_role::sampler) {
        g_sampler_service_ready = true;
        return {};
    }

    if (role != worker_role::tokenizer && !store.metadata_bytes().has_value()) {
        err = "metadata.bin missing in layer store; run install/sync first";
        return {};
    }

    const std::string role_name = worker_role_to_string(role);
    const semantic_runtime_descriptor rt = build_semantic_runtime_descriptor(*manifest);

    if (g_verify_materialization) {
        std::string verr;
        if (!verify_worker_materialization_for_role(
                    store,
                    *manifest,
                    role,
                    cfg.layer_start,
                    cfg.layer_end,
                    verr)) {
            err = "materialization verification failed: " + verr;
            return {};
        }
    }

    const bool force_materialize = body.value("force_materialize", false) ||
            std::getenv("DIST_RUNTIME_FORCE_MATERIALIZE") != nullptr;
    const bool bind_only         = body.value("bind_only", false);

    const runtime_bind_result bind = runtime_bind_worker(
            store, *manifest, role, cfg.layer_start, cfg.layer_end);
    if (!bind.success && !bind.tensors_ready) {
        err = bind.error.empty() ? "layer store bind failed" : bind.error;
        return {};
    }

    if (!force_materialize && bind.cached_gguf_ready && !bind.worker_gguf_path.empty()) {
        const bool cache_ok = role != worker_role::tokenizer ||
                              tokenizer_shell_loadable(bind.worker_gguf_path);
        if (cache_ok) {
            fprintf(stderr,
                    "node_agent: using cached %s role=%s layers=[%d,%d)\n",
                    bind.worker_gguf_path.c_str(),
                    role_name.c_str(),
                    cfg.layer_start,
                    cfg.layer_end);
            return bind.worker_gguf_path;
        }
        fprintf(stderr,
                "node_agent: stale cached %s for role=%s, rematerializing\n",
                bind.worker_gguf_path.c_str(),
                role_name.c_str());
    }

    if (bind_only) {
        if (!bind.tensors_ready) {
            err = bind.error.empty() ? "tensors not ready in layer store" : bind.error;
            return {};
        }
        if (!bind.materialize_required && !bind.worker_gguf_path.empty()) {
            return bind.worker_gguf_path;
        }
        err.clear();
        return {};
    }

    std::string out_path =
            (store.model_root() / ("worker_" + role_name + ".gguf")).string();

    if (role == worker_role::tokenizer) {
        if (!layer_store_materialize_tokenizer_shell(store, *manifest, out_path)) {
            err = "failed to materialize tokenizer shell";
            return {};
        }
    } else if (!materialize_worker_gguf(
                       store,
                       *manifest,
                       rt,
                       role,
                       cfg.layer_start,
                       cfg.layer_end,
                       out_path,
                       err)) {
        err = "failed to assemble GGUF from layer store for role " + role_name + ": " + err;
        return {};
    }

    fprintf(stderr,
            "node_agent: materialized %s role=%s layers=[%d,%d)\n",
            out_path.c_str(),
            role_name.c_str(),
            cfg.layer_start,
            cfg.layer_end);
    g_runtime_stats.materialization_count++;
    g_runtime_stats.materialization_generation++;
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

    dist_debug_load_config();
    dist_debug_set_worker("node_agent", g_node_id);
    split_tcp_init();

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
            { "worker_pid", (int) g_entry_worker.pid },
            { "workers", {
                { "entry", (int) g_entry_worker.pid },
                { "middle", (int) g_middle_worker.pid },
                { "final", (int) g_final_worker.pid },
            }},
            { "runtime_stats", node_runtime_stats_json() },
        }).dump(), "application/json");
    });

    svr.Post("/runtime/tokenizer/configure", [](const httplib::Request & req, httplib::Response & res) {
        json body;
        try {
            body = json::parse(req.body);
        } catch (...) {
            res.status = 400;
            res.set_content(R"({"error":"invalid json"})", "application/json");
            return;
        }
        const std::string path = body.value("worker_gguf", "");
        if (path.empty()) {
            res.status = 400;
            res.set_content(R"({"ok":false,"error":"worker_gguf required"})", "application/json");
            return;
        }
        g_tokenizer_service_gguf = path;
        free_tokenizer_service();
        const bool ready = tokenizer_service_vocab() != nullptr;
        res.set_content(json({
            { "ok", ready },
            { "tokenizer_ready", ready },
        }).dump(), "application/json");
    });

    svr.Post("/runtime/tokenizer/tokenize", [](const httplib::Request & req, httplib::Response & res) {
        json body;
        try {
            body = json::parse(req.body);
        } catch (...) {
            res.status = 400;
            res.set_content(R"({"error":"invalid json"})", "application/json");
            return;
        }
        const std::string prompt = body.value("prompt", "");
        const llama_vocab * vocab = tokenizer_service_vocab();
        if (!vocab) {
            res.status = 503;
            res.set_content(R"({"ok":false,"error":"tokenizer service not configured"})", "application/json");
            return;
        }
        const std::vector<llama_token> toks = split_gen_tokenize(vocab, prompt);
        json tokens = json::array();
        for (const llama_token t : toks) {
            tokens.push_back((int32_t) t);
        }
        res.set_content(json({ { "ok", true }, { "tokens", tokens } }).dump(), "application/json");
    });

    svr.Post("/runtime/tokenizer/detokenize", [](const httplib::Request & req, httplib::Response & res) {
        json body;
        try {
            body = json::parse(req.body);
        } catch (...) {
            res.status = 400;
            res.set_content(R"({"error":"invalid json"})", "application/json");
            return;
        }
        const llama_vocab * vocab = tokenizer_service_vocab();
        if (!vocab || !body.contains("tokens") || !body["tokens"].is_array()) {
            res.status = 400;
            res.set_content(R"({"ok":false,"error":"tokenizer or tokens missing"})", "application/json");
            return;
        }
        std::string text;
        for (const auto & t : body["tokens"]) {
            text += split_gen_token_text(vocab, (llama_token) t.get<int32_t>());
        }
        res.set_content(json({ { "ok", true }, { "text", text } }).dump(), "application/json");
    });

    svr.Post("/runtime/bind", [](const httplib::Request & req, httplib::Response & res) {
        json body;
        try {
            body = json::parse(req.body);
        } catch (...) {
            res.status = 400;
            res.set_content(R"({"error":"invalid json"})", "application/json");
            return;
        }
        dist_configure_req cfg = parse_configure(body);
        if (cfg.model_id.empty()) {
            res.status = 400;
            res.set_content(R"({"ok":false,"error":"model_id required"})", "application/json");
            return;
        }
        layer_store store = get_layer_store(cfg.model_id);
        const auto manifest = store.load_manifest();
        if (!manifest.has_value()) {
            res.status = 404;
            res.set_content(R"({"ok":false,"error":"manifest not found"})", "application/json");
            return;
        }
        worker_role role = dist_role_to_worker_role(cfg.role);
        if (body.contains("runtime_role")) {
            const runtime_role rt = runtime_role_from_string(body.value("runtime_role", ""));
            if (rt == runtime_role::pipeline_stage && cfg.role != DIST_ROLE_UNCONFIGURED) {
                role = dist_role_to_worker_role(cfg.role);
            } else if (rt != runtime_role::unassigned) {
                role = worker_role_from_runtime_role(rt);
            }
        }
        const runtime_bind_result bind = runtime_bind_worker(
                store, *manifest, role, cfg.layer_start, cfg.layer_end);
        res.set_content(json({
            { "ok", bind.success || bind.tensors_ready },
            { "tensors_ready", bind.tensors_ready },
            { "cached_gguf_ready", bind.cached_gguf_ready },
            { "materialize_required", bind.materialize_required },
            { "worker_gguf", bind.worker_gguf_path },
            { "error", bind.error },
        }).dump(), "application/json");
    });

    svr.Post("/runtime/embedding/configure", [](const httplib::Request & req, httplib::Response & res) {
        json body;
        try {
            body = json::parse(req.body);
        } catch (...) {
            res.status = 400;
            res.set_content(R"({"error":"invalid json"})", "application/json");
            return;
        }
        const std::string path = body.value("worker_gguf", "");
        g_embedding_service_gguf = path;
        res.set_content(json({
            { "ok", !path.empty() },
            { "embedding_ready", !path.empty() },
        }).dump(), "application/json");
    });

    svr.Post("/runtime/output/configure", [](const httplib::Request & req, httplib::Response & res) {
        json body;
        try {
            body = json::parse(req.body);
        } catch (...) {
            res.status = 400;
            res.set_content(R"({"error":"invalid json"})", "application/json");
            return;
        }
        const std::string path = body.value("worker_gguf", "");
        g_output_service_gguf = path;
        res.set_content(json({
            { "ok", !path.empty() },
            { "output_ready", !path.empty() },
        }).dump(), "application/json");
    });

    svr.Post("/runtime/sampler/configure", [](const httplib::Request &, httplib::Response & res) {
        g_sampler_service_ready = true;
        res.set_content(json({ { "ok", true }, { "sampler_ready", true } }).dump(), "application/json");
    });

    svr.Post("/runtime/prepare", [](const httplib::Request & req, httplib::Response & res) {
        json body;
        try {
            body = json::parse(req.body);
        } catch (...) {
            res.status = 400;
            res.set_content(R"({"error":"invalid json"})", "application/json");
            return;
        }

        const auto t0 = std::chrono::steady_clock::now();
        dist_configure_req cfg = parse_configure(body);
        std::string err;
        const std::string rt_role = body.value("runtime_role", "");
        const std::string worker_gguf = configure_materialize_worker_gguf(cfg, body, err);
        const bool is_sampler = rt_role == "sampler";
        if (worker_gguf.empty() && !err.empty()) {
            res.status = 500;
            res.set_content(json({
                { "status", "error" },
                { "runtime_ready", false },
                { "error", err },
            }).dump(), "application/json");
            return;
        }
        if (worker_gguf.empty() && !is_sampler) {
            res.status = 500;
            res.set_content(json({
                { "status", "error" },
                { "runtime_ready", false },
                { "error", "prepare produced no worker artifact" },
            }).dump(), "application/json");
            return;
        }

        bool tokenizer_ready = false;
        if (rt_role == "tokenizer") {
            g_tokenizer_service_gguf = worker_gguf;
            free_tokenizer_service();
            tokenizer_ready = tokenizer_service_vocab() != nullptr;
            if (!tokenizer_ready) {
                res.status = 500;
                res.set_content(json({
                    { "status", "error" },
                    { "runtime_ready", false },
                    { "error", "tokenizer shell failed to load" },
                    { "worker_gguf", worker_gguf },
                }).dump(), "application/json");
                return;
            }
        } else if (rt_role == "embedding") {
            g_embedding_service_gguf = worker_gguf;
        } else if (rt_role == "output_head") {
            g_output_service_gguf = worker_gguf;
        } else if (is_sampler) {
            g_sampler_service_ready = true;
        } else if (cfg.role == DIST_ROLE_ENTRY) {
            g_entry_worker_gguf = worker_gguf;
            free_entry_tokenizer();
            tokenizer_ready = entry_node_vocab() != nullptr;
        }

        layer_store store = get_layer_store(cfg.model_id);
        const auto manifest = store.load_manifest();
        worker_role bind_role = dist_role_to_worker_role(cfg.role);
        if (!rt_role.empty()) {
            const runtime_role rt = runtime_role_from_string(rt_role);
            if (rt == runtime_role::pipeline_stage) {
                bind_role = dist_role_to_worker_role(cfg.role);
            } else if (rt != runtime_role::unassigned) {
                bind_role = worker_role_from_runtime_role(rt);
            }
        }
        bool tensors_ready = false;
        bool cached_gguf   = false;
        if (manifest.has_value()) {
            const runtime_bind_result bind = runtime_bind_worker(
                    store, *manifest, bind_role, cfg.layer_start, cfg.layer_end);
            tensors_ready = bind.tensors_ready;
            cached_gguf   = bind.cached_gguf_ready;
        }

        const double ms = std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - t0).count();
        res.set_content(json({
            { "status", "ok" },
            { "runtime_ready", true },
            { "worker_gguf", worker_gguf },
            { "tokenizer_ready", tokenizer_ready },
            { "tensors_ready", tensors_ready },
            { "cached_gguf", cached_gguf },
            { "materialization_time_ms", ms },
        }).dump(), "application/json");
    });

    svr.Post("/configure", [agent_bin](const httplib::Request & req, httplib::Response & res) {
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
        const bool skip_materialize = body.value("skip_materialize", false);

        if (dist_debug_enabled() && !cfg.session_id.empty()) {
            dist_debug_set_session(cfg.session_id);
        }

        if (skip_materialize) {
            worker_model = body.value("worker_gguf", "");
            if (worker_model.empty()) {
                res.status = 400;
                res.set_content(json({
                    { "ok", false },
                    { "error", "worker_gguf required when skip_materialize=true" },
                }).dump(), "application/json");
                return;
            }
        } else if (!cfg.model_id.empty()) {
            const std::string materialized = configure_materialize_worker_gguf(cfg, body, err);
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
        g_runtime_stats.configure_count++;
        g_runtime_stats.runtime_load_count++;

        if (cfg.role == DIST_ROLE_ENTRY) {
            g_pipeline_ctrl_port = cfg.ctrl_port;
            g_pipeline_layer_end   = cfg.layer_end;
            g_entry_worker_gguf    = worker_model;
        }

        res.set_content(json({
            { "ok", true },
            { "role", dist_role_name(cfg.role) },
            { "worker_pid", (int) (worker_proc_slot(cfg.role) ? worker_proc_slot(cfg.role)->pid : 0) },
        }).dump(), "application/json");
    });

    svr.Post("/pipeline/generate", [](const httplib::Request & req, httplib::Response & res) {
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

        const std::string prompt_text = body.value("prompt", "");
        if (prompt_tokens.empty() && !prompt_text.empty()) {
            res.status = 400;
            res.set_content(R"({"ok":false,"error":"prompt_tokens required; use tokenizer service"})", "application/json");
            return;
        }

        if (prompt_tokens.empty()) {
            res.status = 400;
            res.set_content(R"({"ok":false,"error":"prompt or prompt_tokens required"})", "application/json");
            return;
        }
        if (layer_end <= 0) {
            res.status = 503;
            res.set_content(R"({"ok":false,"error":"pipeline not configured"})", "application/json");
            return;
        }

        std::vector<int32_t> out_tokens;
        std::string err;
        json timing = json::object();
        if (!run_local_pipeline_generate(prompt_tokens, max_tokens, layer_end, out_tokens, err, &timing)) {
            res.status = 500;
            res.set_content(json({ { "ok", false }, { "error", err }, { "tokens", json::array() } }).dump(),
                    "application/json");
            return;
        }

        json tokens_json = json::array();
        std::string text;
        for (const int32_t t : out_tokens) {
            tokens_json.push_back(t);
        }
        g_runtime_stats.pipeline_generate_count++;
        res.set_content(json({
            { "ok", true },
            { "tokens", tokens_json },
            { "text", text },
            { "count", out_tokens.size() },
            { "timing", timing },
        }).dump(), "application/json");
    });

    svr.Post("/shutdown", [](const httplib::Request &, httplib::Response & res) {
        stop_all_workers();
        free_entry_tokenizer();
        free_tokenizer_service();
        g_entry_worker_gguf.clear();
        g_tokenizer_service_gguf.clear();
        g_embedding_service_gguf.clear();
        g_output_service_gguf.clear();
        g_sampler_service_ready = false;
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

            for (const layer_store::blob_tensor_info & blob : store.list_blob_tensors()) {
                std::string state = "READY";
                const std::string checksum = blob.checksum.empty()
                        ? ("manifest:tensor:" + blob.tensor_name)
                        : blob.checksum;
                if (!store.verify_blob_tensor(blob.blob_id, blob.tensor_name, checksum)) {
                    state = "CORRUPTED";
                }

                layers_json.push_back({
                    { "layer", -1 },
                    { "layer_index", -1 },
                    { "blob_id", blob.blob_id },
                    { "tensor_name", blob.tensor_name },
                    { "node", g_node_id },
                    { "node_id", g_node_id },
                    { "device", device },
                    { "size_bytes", blob.size_bytes },
                    { "checksum", checksum },
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
    // POST /models/{model_id}/reset - Clear local layer store bytes.
    svr.Get(R"(/models/([^/]+)/workers/entry)", [](const httplib::Request & req, httplib::Response & res) {
        const std::string model_id = req.matches[1];
        layer_store store = get_layer_store(model_id);
        const std::filesystem::path path = store.model_root() / "worker_entry.gguf";
        std::error_code ec;
        if (!std::filesystem::exists(path, ec)) {
            res.status = 404;
            res.set_content(json({ { "error", "worker_entry.gguf not found" } }).dump(), "application/json");
            return;
        }
        std::ifstream in(path, std::ios::binary);
        if (!in) {
            res.status = 500;
            res.set_content(json({ { "error", "failed to read worker_entry.gguf" } }).dump(), "application/json");
            return;
        }
        std::string body((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
        res.set_content(std::move(body), "application/octet-stream");
    });

    svr.Post(R"(/models/([^/]+)/reset)", [](const httplib::Request & req, httplib::Response & res) {
        const std::string model_id = req.matches[1];

        json body;
        try {
            body = req.body.empty() ? json::object() : json::parse(req.body);
        } catch (...) {
            body = json::object();
        }
        const bool keep_manifest = body.value("keep_manifest", true);

        layer_store store = get_layer_store(model_id);
        if (!store.clear_model_storage(keep_manifest)) {
            res.status = 500;
            res.set_content(json({ { "ok", false }, { "error", "clear failed" } }).dump(), "application/json");
            return;
        }

        res.set_content(json({
            { "ok", true },
            { "model_id", model_id },
            { "keep_manifest", keep_manifest },
            { "node_id", g_node_id },
        }).dump(), "application/json");
    });

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
        stop_all_workers();
        return 1;
    }

    stop_all_workers();
    return 0;
}
