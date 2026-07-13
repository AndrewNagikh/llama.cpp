#include "perf_trace.h"

#include "perf_gpu_sampler.h"
#include "perf_ggml.h"

#include <chrono>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <mutex>
#include <sstream>

namespace {

using clock = std::chrono::steady_clock;

static perf_trace_config g_cfg{};
static bool              g_cfg_loaded = false;

static std::mutex        g_mu;
static std::ofstream     g_out;
static std::string       g_out_path;
static int64_t           g_epoch_us = 0;

static std::string g_trace_id;
static std::string g_phase = "decode";
static int32_t     g_token_idx = -1;
static int32_t     g_wave_id   = PERF_WAVE_ID_NONE;
static std::string g_output_dir;
static std::string g_base_trace_dir;

static bool env_truthy(const char * name) {
    const char * v = std::getenv(name);
    if (v == nullptr || v[0] == '\0') {
        return false;
    }
    return std::strcmp(v, "0") != 0 &&
           std::strcmp(v, "false") != 0 &&
           std::strcmp(v, "FALSE") != 0;
}

static std::string json_escape(const std::string & s) {
    std::ostringstream os;
    os << '"';
    for (char c : s) {
        switch (c) {
            case '"':  os << "\\\""; break;
            case '\\': os << "\\\\"; break;
            case '\n': os << "\\n";  break;
            default:   os << c;      break;
        }
    }
    os << '"';
    return os.str();
}

static void ensure_trace_dir(const std::string & dir) {
    std::error_code ec;
    std::filesystem::create_directories(dir, ec);
}

static void apply_output_dir_from_context(const std::string & output_subdir) {
    if (output_subdir.empty()) {
        g_output_dir = g_base_trace_dir;
        return;
    }
    g_output_dir = g_base_trace_dir + "/" + output_subdir;
}

static int32_t derive_wave_id(const std::string & phase, const int32_t token_idx, const int32_t wave_id_in) {
    if (wave_id_in >= 0) {
        return wave_id_in;
    }
    if (wave_id_in == PERF_WAVE_ID_NONE) {
        return PERF_WAVE_ID_NONE;
    }
    if (phase == "ttft" || phase == "prefill") {
        return 0;
    }
    if (token_idx >= 0) {
        return token_idx + 1;
    }
    return PERF_WAVE_ID_NONE;
}

static void write_active_context_file() {
    if (!perf_trace_enabled() || g_trace_id.empty()) {
        return;
    }
    ensure_trace_dir(g_base_trace_dir);
    const std::string path = g_base_trace_dir + "/active_context.json";
    std::string output_subdir;
    if (!g_output_dir.empty() && g_output_dir.rfind(g_base_trace_dir, 0) == 0) {
        output_subdir = g_output_dir.substr(g_base_trace_dir.size());
        if (!output_subdir.empty() && output_subdir[0] == '/') {
            output_subdir = output_subdir.substr(1);
        }
    }
    std::ofstream f(path, std::ios::trunc);
    if (!f) {
        return;
    }
    f << "{"
      << "\"trace_id\":" << json_escape(g_trace_id) << ","
      << "\"phase\":" << json_escape(g_phase) << ","
      << "\"token_idx\":" << g_token_idx << ","
      << "\"WaveID\":" << g_wave_id << ","
      << "\"output_subdir\":" << json_escape(output_subdir)
      << "}";
    if (!g_output_dir.empty()) {
        ensure_trace_dir(g_output_dir);
    }
}

static void parse_active_context_file() {
    if (g_base_trace_dir.empty()) {
        return;
    }
    const std::string path = g_base_trace_dir + "/active_context.json";
    std::ifstream f(path);
    if (!f) {
        return;
    }
    std::string content((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());

    auto extract_string = [&](const char * key, std::string & out) {
        const std::string needle = std::string("\"") + key + "\":\"";
        const size_t pos = content.find(needle);
        if (pos == std::string::npos) {
            return;
        }
        const size_t start = pos + needle.size();
        const size_t end = content.find('"', start);
        if (end == std::string::npos) {
            return;
        }
        out = content.substr(start, end - start);
    };

    auto extract_int = [&](const char * key, int32_t & out) {
        const std::string needle = std::string("\"") + key + "\":";
        const size_t pos = content.find(needle);
        if (pos == std::string::npos) {
            return;
        }
        out = std::atoi(content.c_str() + pos + needle.size());
    };

    extract_string("trace_id", g_trace_id);
    extract_string("phase", g_phase);
    extract_int("token_idx", g_token_idx);
    int32_t parsed_wave = PERF_WAVE_ID_NONE;
    extract_int("WaveID", parsed_wave);
    if (parsed_wave >= 0) {
        g_wave_id = parsed_wave;
    }
    std::string output_subdir;
    extract_string("output_subdir", output_subdir);
    apply_output_dir_from_context(output_subdir);
}

static void ensure_open() {
    if (!perf_trace_enabled()) {
        return;
    }
    if (g_out.is_open()) {
        return;
    }
    if (g_output_dir.empty()) {
        return;
    }
    ensure_trace_dir(g_output_dir);
    const std::string comp = g_cfg.component.empty() ? "component" : g_cfg.component;
    const std::string nid  = g_cfg.node_id.empty() ? "node" : g_cfg.node_id;
    const std::string tid  = g_trace_id.empty() ? "notrace" : g_trace_id;
    g_out_path = g_output_dir + "/" + tid + "_" + nid + "_" + comp + ".jsonl";
    g_out.open(g_out_path, std::ios::app);
}

static void write_event_line(const std::string & line) {
    if (!perf_trace_enabled()) {
        return;
    }
    std::lock_guard<std::mutex> lock(g_mu);
    ensure_open();
    if (!g_out) {
        return;
    }
    g_out << line << '\n';
    g_out.flush();
}

static void write_event(
        const char * kind,
        const char * event,
        perf_category category,
        const char * stage,
        int32_t token_idx,
        int64_t ts_us,
        int64_t dur_us,
        const char * attrs_json) {
    if (!perf_trace_enabled()) {
        return;
    }
    std::ostringstream os;
    os << "{"
       << "\"kind\":" << json_escape(kind) << ","
       << "\"trace_id\":" << json_escape(g_trace_id) << ","
       << "\"phase\":" << json_escape(g_phase) << ","
       << "\"token_idx\":" << token_idx << ","
       << "\"WaveID\":" << g_wave_id << ","
       << "\"stage\":" << json_escape(stage ? stage : "") << ","
       << "\"node_id\":" << json_escape(g_cfg.node_id) << ","
       << "\"component\":" << json_escape(g_cfg.component) << ","
       << "\"event\":" << json_escape(event) << ","
       << "\"category\":" << json_escape(perf_category_name(category)) << ","
       << "\"ts_us\":" << ts_us;
    if (dur_us >= 0) {
        os << ",\"dur_us\":" << dur_us;
    }
    if (attrs_json != nullptr && attrs_json[0] != '\0') {
        os << ",\"attrs\":" << attrs_json;
    }
    os << "}";
    write_event_line(os.str());
}

} // namespace

const char * perf_category_name(const perf_category category) {
    switch (category) {
        case perf_category::COMPUTE:        return "COMPUTE";
        case perf_category::WAIT:           return "WAIT";
        case perf_category::NETWORK:        return "NETWORK";
        case perf_category::SERIALIZATION:  return "SERIALIZATION";
        case perf_category::SAMPLING:       return "SAMPLING";
        case perf_category::IDLE:           return "IDLE";
        case perf_category::INSTALL_REUSE:  return "INSTALL_REUSE";
        case perf_category::SESSION:        return "SESSION";
        case perf_category::TTFT:           return "TTFT";
        case perf_category::GPU:            return "GPU";
        default:                            return "UNKNOWN";
    }
}

void perf_trace_load_config() {
    if (g_cfg_loaded) {
        return;
    }
    g_cfg.enabled = env_truthy("DIST_PERF_TRACE");
    if (const char * dir = std::getenv("DIST_PERF_TRACE_DIR")) {
        g_cfg.trace_dir = dir;
    } else if (const char * models = std::getenv("MODELS_DIR")) {
        g_cfg.trace_dir = std::string(models) + "/perf_trace";
    } else {
        g_cfg.trace_dir = "/tmp/dist_perf_trace";
    }
    if (const char * nid = std::getenv("DIST_NODE_ID")) {
        g_cfg.node_id = nid;
    }
    if (const char * comp = std::getenv("DIST_PERF_COMPONENT")) {
        g_cfg.component = comp;
    }
    g_cfg_loaded = true;
    g_base_trace_dir = g_cfg.trace_dir;
    perf_trace_register_ggml_hooks();
}

void perf_trace_reload_config() {
    g_cfg_loaded = false;
    perf_trace_load_config();
}

bool perf_trace_enabled() {
    perf_trace_load_config();
    return g_cfg.enabled;
}

const perf_trace_config & perf_trace_config_get() {
    perf_trace_load_config();
    return g_cfg;
}

void perf_trace_set_node_id(const std::string & node_id) {
    perf_trace_load_config();
    g_cfg.node_id = node_id;
}

void perf_trace_set_component(const std::string & component) {
    perf_trace_load_config();
    g_cfg.component = component;
}

uint64_t perf_now_us() {
    return static_cast<uint64_t>(
            std::chrono::duration_cast<std::chrono::microseconds>(
                    clock::now().time_since_epoch()).count());
}

int64_t perf_trace_epoch_us() {
    return g_epoch_us;
}

void perf_trace_set_context(
        const std::string & trace_id,
        const std::string & phase,
        const int32_t token_idx,
        const int32_t wave_id) {
    g_trace_id   = trace_id;
    g_phase      = phase.empty() ? "decode" : phase;
    g_token_idx  = token_idx;
    g_wave_id    = derive_wave_id(g_phase, token_idx, wave_id);
    if (!trace_id.empty()) {
        g_output_dir = g_base_trace_dir + "/" + trace_id;
        if (!g_phase.empty()) {
            g_output_dir += "/" + g_phase;
        }
        ensure_trace_dir(g_output_dir);
    }
    write_active_context_file();
}

void perf_trace_set_wave_id(const int32_t wave_id) {
    g_wave_id = wave_id;
}

int32_t perf_trace_get_wave_id() {
    return g_wave_id;
}

int32_t perf_trace_wave_id_from_step(const char * phase, const int32_t debug_step) {
    if (phase == nullptr) {
        return PERF_WAVE_ID_NONE;
    }
    if (std::strcmp(phase, "prefill") == 0) {
        return 0;
    }
    if (std::strcmp(phase, "decode") == 0 && debug_step > 0) {
        return debug_step;
    }
    return PERF_WAVE_ID_NONE;
}

int32_t perf_trace_derive_wave_id(const std::string & phase, const int32_t token_idx) {
    return derive_wave_id(phase, token_idx, PERF_WAVE_ID_AUTO);
}

static void parse_decode_context_file() {
    if (g_base_trace_dir.empty()) {
        return;
    }
    const std::string path = g_base_trace_dir + "/decode_context.json";
    std::ifstream f(path);
    if (!f) {
        return;
    }
    std::string content((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());

    auto extract_string = [&](const char * key, std::string & out) {
        const std::string needle = std::string("\"") + key + "\":\"";
        const size_t pos = content.find(needle);
        if (pos == std::string::npos) {
            return;
        }
        const size_t start = pos + needle.size();
        const size_t end = content.find('"', start);
        if (end == std::string::npos) {
            return;
        }
        out = content.substr(start, end - start);
    };

    std::string trace_id;
    extract_string("trace_id", trace_id);
    if (!trace_id.empty()) {
        g_trace_id = trace_id;
        g_phase    = "decode";
    }
}

void perf_trace_write_decode_context(const std::string & trace_id) {
    if (trace_id.empty()) {
        return;
    }
    perf_trace_load_config();
    if (g_base_trace_dir.empty() && !g_cfg.trace_dir.empty()) {
        g_base_trace_dir = g_cfg.trace_dir;
    }
    if (g_base_trace_dir.empty()) {
        return;
    }
    ensure_trace_dir(g_base_trace_dir);
    const std::string path = g_base_trace_dir + "/decode_context.json";
    std::ofstream f(path, std::ios::trunc);
    if (!f) {
        return;
    }
    f << "{"
      << "\"trace_id\":" << json_escape(trace_id) << ","
      << "\"phase\":\"decode\""
      << "}";
    g_trace_id = trace_id;
    g_phase    = "decode";
    write_active_context_file();
}

void perf_trace_refresh_context() {
    if (!perf_trace_enabled()) {
        return;
    }
    parse_active_context_file();
    parse_decode_context_file();
}

void perf_trace_ensure_decode_context(const int32_t token_idx, const int32_t wave_id) {
    if (!perf_trace_enabled()) {
        return;
    }
    perf_trace_refresh_context();
    std::string trace_id;
    std::string phase;
    int32_t     cur_tok = -1;
    if (!perf_trace_get_context(trace_id, phase, cur_tok) || trace_id.empty()) {
        return;
    }
    perf_trace_set_context(trace_id, "decode", token_idx, wave_id);
}

bool perf_trace_get_context(std::string & trace_id, std::string & phase, int32_t & token_idx) {
    trace_id  = g_trace_id;
    phase     = g_phase;
    token_idx = g_token_idx;
    return !g_trace_id.empty();
}

bool perf_trace_get_context(
        std::string & trace_id,
        std::string & phase,
        int32_t & token_idx,
        int32_t & wave_id) {
    trace_id  = g_trace_id;
    phase     = g_phase;
    token_idx = g_token_idx;
    wave_id   = g_wave_id;
    return !g_trace_id.empty();
}

bool perf_trace_has_active_context() {
    perf_trace_load_config();
    return g_cfg.enabled && !g_trace_id.empty();
}

void perf_trace_begin_generate(const std::string & trace_id, const std::string & subdir) {
    if (!perf_trace_enabled()) {
        return;
    }
    perf_trace_load_config();
    g_base_trace_dir = g_cfg.trace_dir;

    g_epoch_us  = static_cast<int64_t>(perf_now_us());
    g_trace_id  = trace_id;
    g_phase     = subdir.empty() ? "decode" : subdir;
    g_token_idx = -1;
    g_wave_id   = PERF_WAVE_ID_NONE;

    g_output_dir = g_base_trace_dir + "/" + trace_id;
    if (!subdir.empty()) {
        g_output_dir += "/" + subdir;
    }
    ensure_trace_dir(g_output_dir);
    write_active_context_file();

    perf_emit_instant("GENERATE_BEGIN", perf_category::UNKNOWN, "orchestrator", -1, nullptr);
    perf_gpu_poll_start();
}

void perf_trace_end_generate() {
    if (!perf_trace_enabled()) {
        return;
    }
    perf_emit_instant("GENERATE_END", perf_category::UNKNOWN, "orchestrator", g_token_idx, nullptr);
    perf_gpu_poll_stop();
    std::lock_guard<std::mutex> lock(g_mu);
    if (g_out.is_open()) {
        g_out.close();
    }
    g_out_path.clear();
    g_output_dir.clear();
    g_trace_id.clear();
}

void perf_trace_begin_install(const std::string & trace_id, const std::string & subdir) {
    if (!perf_trace_enabled()) {
        return;
    }
    perf_trace_load_config();
    g_base_trace_dir = g_cfg.trace_dir;

    g_epoch_us  = static_cast<int64_t>(perf_now_us());
    g_trace_id  = trace_id;
    g_phase     = "install";
    g_token_idx = -1;

    g_output_dir = g_base_trace_dir + "/" + trace_id;
    if (!subdir.empty()) {
        g_output_dir += "/" + subdir;
    }
    ensure_trace_dir(g_output_dir);
    write_active_context_file();

    perf_emit_install_instant("INSTALL_BEGIN", "plan", nullptr, nullptr, 0, nullptr);
    perf_gpu_poll_start();
}

void perf_trace_end_install() {
    if (!perf_trace_enabled()) {
        return;
    }
    perf_emit_install_instant("INSTALL_END", "plan", nullptr, nullptr, 0, nullptr);
    perf_gpu_poll_stop();
    std::lock_guard<std::mutex> lock(g_mu);
    if (g_out.is_open()) {
        g_out.close();
    }
    g_out_path.clear();
    g_output_dir.clear();
    g_trace_id.clear();
}

static void write_install_event(
        const char * kind,
        const char * event,
        const char * sub,
        const char * blob_id,
        const char * node_id,
        uint64_t bytes,
        int64_t ts_us,
        int64_t dur_us,
        const char * attrs_json) {
    if (!perf_trace_enabled()) {
        return;
    }
    std::ostringstream os;
    os << "{"
       << "\"kind\":" << json_escape(kind) << ","
       << "\"trace_id\":" << json_escape(g_trace_id) << ","
       << "\"phase\":" << json_escape(g_phase.empty() ? "install" : g_phase) << ","
       << "\"token_idx\":-1,"
       << "\"stage\":\"install\","
       << "\"node_id\":" << json_escape(node_id ? node_id : g_cfg.node_id) << ","
       << "\"component\":" << json_escape(g_cfg.component) << ","
       << "\"event\":" << json_escape(event) << ","
       << "\"category\":" << json_escape(perf_category_name(perf_category::INSTALL_REUSE)) << ","
       << "\"ts_us\":" << ts_us;
    if (dur_us >= 0) {
        os << ",\"dur_us\":" << dur_us;
    }
    os << ",\"attrs\":{"
       << "\"sub\":" << json_escape(sub ? sub : "") << ","
       << "\"blob_id\":" << json_escape(blob_id ? blob_id : "") << ","
       << "\"bytes\":" << bytes;
    if (attrs_json != nullptr && attrs_json[0] != '\0') {
        os << ",\"extra\":" << attrs_json;
    }
    os << "}}";
    write_event_line(os.str());
}

void perf_emit_install_span(
        const char * event,
        const char * sub,
        const char * blob_id,
        const char * node_id,
        const uint64_t bytes,
        const int64_t dur_us) {
    write_install_event(
            "span",
            event,
            sub,
            blob_id,
            node_id,
            bytes,
            static_cast<int64_t>(perf_now_us()),
            dur_us,
            nullptr);
}

void perf_emit_install_instant(
        const char * event,
        const char * sub,
        const char * blob_id,
        const char * node_id,
        const uint64_t bytes,
        const char * attrs_json) {
    write_install_event(
            "instant",
            event,
            sub,
            blob_id,
            node_id,
            bytes,
            static_cast<int64_t>(perf_now_us()),
            -1,
            attrs_json);
}

perf_install_span::perf_install_span(
        const char * event,
        const char * sub,
        const char * blob_id,
        const char * node_id)
    : event_(event),
      sub_(sub),
      blob_id_(blob_id),
      node_id_(node_id) {
    if (!perf_trace_enabled()) {
        return;
    }
    perf_trace_refresh_context();
    active_ = true;
    t0_us_  = static_cast<int64_t>(perf_now_us());
}

perf_install_span::~perf_install_span() {
    if (!active_) {
        return;
    }
    const int64_t dur = static_cast<int64_t>(perf_now_us()) - t0_us_;
    perf_emit_install_span(event_, sub_, blob_id_, node_id_, bytes_, dur);
}

void perf_trace_begin_session(const std::string & trace_id, const std::string & subdir) {
    if (!perf_trace_enabled()) {
        return;
    }
    perf_trace_load_config();
    g_base_trace_dir = g_cfg.trace_dir;

    g_epoch_us  = static_cast<int64_t>(perf_now_us());
    g_trace_id  = trace_id;
    g_phase     = "session_create";
    g_token_idx = -1;

    g_output_dir = g_base_trace_dir + "/" + trace_id;
    if (!subdir.empty()) {
        g_output_dir += "/" + subdir;
    }
    ensure_trace_dir(g_output_dir);
    write_active_context_file();

    perf_emit_session_instant("SESSION_BEGIN", "orchestrator", "orchestrator", nullptr);
    perf_gpu_poll_start();
}

void perf_trace_end_session() {
    if (!perf_trace_enabled()) {
        return;
    }
    perf_emit_session_instant("SESSION_END", "orchestrator", "orchestrator", nullptr);
    perf_gpu_poll_stop();
    std::lock_guard<std::mutex> lock(g_mu);
    if (g_out.is_open()) {
        g_out.close();
    }
    g_out_path.clear();
    g_output_dir.clear();
    g_trace_id.clear();
}

static void write_session_event(
        const char * kind,
        const char * event,
        const char * node_id,
        const char * role,
        int64_t ts_us,
        int64_t dur_us,
        const char * attrs_json) {
    if (!perf_trace_enabled()) {
        return;
    }
    std::ostringstream os;
    os << "{"
       << "\"kind\":" << json_escape(kind) << ","
       << "\"trace_id\":" << json_escape(g_trace_id) << ","
       << "\"phase\":" << json_escape(g_phase.empty() ? "session_create" : g_phase) << ","
       << "\"token_idx\":-1,"
       << "\"stage\":" << json_escape(role ? role : "") << ","
       << "\"node_id\":" << json_escape(node_id ? node_id : g_cfg.node_id) << ","
       << "\"component\":" << json_escape(g_cfg.component) << ","
       << "\"event\":" << json_escape(event) << ","
       << "\"category\":" << json_escape(perf_category_name(perf_category::SESSION)) << ","
       << "\"ts_us\":" << ts_us;
    if (dur_us >= 0) {
        os << ",\"dur_us\":" << dur_us;
    }
    os << ",\"attrs\":{"
       << "\"role\":" << json_escape(role ? role : "");
    if (attrs_json != nullptr && attrs_json[0] != '\0') {
        os << ",\"extra\":" << attrs_json;
    }
    os << "}}";
    write_event_line(os.str());
}

void perf_emit_session_span(
        const char * event,
        const char * node_id,
        const char * role,
        const int64_t dur_us,
        const char * attrs_json) {
    write_session_event(
            "span",
            event,
            node_id,
            role,
            static_cast<int64_t>(perf_now_us()),
            dur_us,
            attrs_json);
}

void perf_emit_session_instant(
        const char * event,
        const char * node_id,
        const char * role,
        const char * attrs_json) {
    write_session_event(
            "instant",
            event,
            node_id,
            role,
            static_cast<int64_t>(perf_now_us()),
            -1,
            attrs_json);
}

perf_session_span::perf_session_span(
        const char * event,
        const char * node_id,
        const char * role)
    : event_(event),
      node_id_(node_id),
      role_(role) {
    if (!perf_trace_enabled()) {
        return;
    }
    perf_trace_refresh_context();
    active_ = true;
    t0_us_  = static_cast<int64_t>(perf_now_us());
}

perf_session_span::~perf_session_span() {
    if (!active_) {
        return;
    }
    const int64_t dur = static_cast<int64_t>(perf_now_us()) - t0_us_;
    perf_emit_session_span(event_, node_id_, role_, dur, nullptr);
}

static void write_ttft_event(
        const char * kind,
        const char * event,
        const char * stage,
        int64_t ts_us,
        int64_t dur_us,
        const char * attrs_json) {
    if (!perf_trace_enabled()) {
        return;
    }
    std::ostringstream os;
    os << "{"
       << "\"kind\":" << json_escape(kind) << ","
       << "\"trace_id\":" << json_escape(g_trace_id) << ","
       << "\"phase\":" << json_escape(g_phase.empty() ? "ttft" : g_phase) << ","
       << "\"token_idx\":-1,"
       << "\"stage\":" << json_escape(stage ? stage : "") << ","
       << "\"node_id\":" << json_escape(g_cfg.node_id) << ","
       << "\"component\":" << json_escape(g_cfg.component) << ","
       << "\"event\":" << json_escape(event) << ","
       << "\"category\":" << json_escape(perf_category_name(perf_category::TTFT)) << ","
       << "\"ts_us\":" << ts_us;
    if (dur_us >= 0) {
        os << ",\"dur_us\":" << dur_us;
    }
    if (attrs_json != nullptr && attrs_json[0] != '\0') {
        os << ",\"attrs\":" << attrs_json;
    } else {
        os << ",\"attrs\":{}";
    }
    os << "}";
    write_event_line(os.str());
}

void perf_emit_ttft_span(
        const char * event,
        const char * stage,
        const int64_t dur_us,
        const char * attrs_json) {
    write_ttft_event(
            "span",
            event,
            stage,
            static_cast<int64_t>(perf_now_us()),
            dur_us,
            attrs_json);
}

void perf_emit_ttft_instant(
        const char * event,
        const char * stage,
        const char * attrs_json) {
    write_ttft_event(
            "instant",
            event,
            stage,
            static_cast<int64_t>(perf_now_us()),
            -1,
            attrs_json);
}

perf_ttft_span::perf_ttft_span(const char * event, const char * stage)
    : event_(event),
      stage_(stage) {
    if (!perf_trace_enabled()) {
        return;
    }
    perf_trace_refresh_context();
    active_ = true;
    t0_us_  = static_cast<int64_t>(perf_now_us());
}

perf_ttft_span::~perf_ttft_span() {
    if (!active_) {
        return;
    }
    const int64_t dur = static_cast<int64_t>(perf_now_us()) - t0_us_;
    perf_emit_ttft_span(event_, stage_, dur, nullptr);
}

void perf_trace_begin_ttft(const std::string & trace_id, const std::string & subdir) {
    if (!perf_trace_enabled()) {
        return;
    }
    perf_trace_load_config();
    g_base_trace_dir = g_cfg.trace_dir;

    g_epoch_us  = static_cast<int64_t>(perf_now_us());
    g_trace_id  = trace_id;
    g_phase     = "ttft";
    g_token_idx = -1;

    g_output_dir = g_base_trace_dir + "/" + trace_id;
    if (!subdir.empty()) {
        g_output_dir += "/" + subdir;
    }
    ensure_trace_dir(g_output_dir);
    write_active_context_file();

    perf_emit_ttft_instant("TTFT_BEGIN", "orchestrator", nullptr);
    perf_gpu_poll_start();
}

void perf_trace_end_ttft() {
    if (!perf_trace_enabled()) {
        return;
    }
    perf_emit_ttft_instant("TTFT_END", "orchestrator", nullptr);
    perf_gpu_poll_stop();
    std::lock_guard<std::mutex> lock(g_mu);
    if (g_out.is_open()) {
        g_out.close();
    }
    g_out_path.clear();
    g_output_dir.clear();
}

void perf_emit_span(
        const char * event,
        const perf_category category,
        const char * stage,
        const int32_t token_idx,
        const int64_t dur_us,
        const char * attrs_json) {
    write_event("span", event, category, stage, token_idx, static_cast<int64_t>(perf_now_us()), dur_us, attrs_json);
}

void perf_emit_instant(
        const char * event,
        const perf_category category,
        const char * stage,
        const int32_t token_idx,
        const char * attrs_json) {
    write_event("instant", event, category, stage, token_idx, static_cast<int64_t>(perf_now_us()), -1, attrs_json);
}

void perf_emit_hidden_transfer(
        const char * stage,
        const int32_t token_idx,
        const char * link,
        const int32_t payload_bytes,
        const int64_t serialize_us,
        const int64_t send_us,
        const int64_t receive_us,
        const int64_t deserialize_us) {
    const int64_t total_us = serialize_us + send_us + receive_us + deserialize_us;
    char attrs[512];
    std::snprintf(
            attrs,
            sizeof(attrs),
            "{\"link\":\"%s\",\"payload_bytes\":%d,"
            "\"serialize_us\":%lld,\"send_us\":%lld,"
            "\"receive_us\":%lld,\"deserialize_us\":%lld,\"total_us\":%lld}",
            link ? link : "",
            payload_bytes,
            (long long) serialize_us,
            (long long) send_us,
            (long long) receive_us,
            (long long) deserialize_us,
            (long long) total_us);
    perf_emit_span("HIDDEN_TRANSFER", perf_category::NETWORK, stage, token_idx, total_us, attrs);
}

void perf_emit_queue_depth(const char * stage, const int32_t token_idx, const int32_t depth) {
    char attrs[64];
    std::snprintf(attrs, sizeof(attrs), "{\"depth\":%d}", depth);
    perf_emit_instant("QUEUE_DEPTH", perf_category::WAIT, stage, token_idx, attrs);
}

void perf_emit_gpu_sample(
        const char * backend,
        const float gpu_util_pct,
        const float gpu_mem_used_mb,
        const char * attrs_json) {
    std::ostringstream merged;
    merged << "{\"backend\":" << json_escape(backend ? backend : "cpu")
           << ",\"gpu_util_pct\":" << gpu_util_pct
           << ",\"gpu_mem_used_mb\":" << gpu_mem_used_mb;
    if (attrs_json != nullptr && attrs_json[0] == '{') {
        const size_t len = std::strlen(attrs_json);
        if (len > 2) {
            merged << "," << std::string(attrs_json + 1, len - 2);
        }
    }
    merged << "}";
    write_event(
            "sample",
            "GPU_SAMPLE",
            perf_category::GPU,
            "node",
            g_token_idx,
            static_cast<int64_t>(perf_now_us()),
            -1,
            merged.str().c_str());
}

perf_span::perf_span(
        const char * begin_event,
        const char * end_event,
        const perf_category category,
        const char * stage)
    : begin_event_(begin_event),
      end_event_(end_event),
      category_(category),
      stage_(stage) {
    if (!perf_trace_enabled()) {
        return;
    }
    perf_trace_refresh_context();
    active_ = true;
    t0_us_  = static_cast<int64_t>(perf_now_us());
    perf_emit_instant(begin_event_, category_, stage_, token_idx_, nullptr);
}

perf_span::~perf_span() {
    if (!active_) {
        return;
    }
    const int64_t dur = static_cast<int64_t>(perf_now_us()) - t0_us_;
    perf_emit_span(end_event_, category_, stage_, token_idx_, dur, nullptr);
}

const char * perf_trace_output_dir() {
    return g_output_dir.empty() ? nullptr : g_output_dir.c_str();
}

std::string perf_make_trace_id(const int seq) {
    char buf[32];
    std::snprintf(buf, sizeof(buf), "trace-%06d", seq);
    return std::string(buf);
}

std::string perf_make_install_trace_id(const std::string & run_id, const std::string & model_id) {
    return "install-" + run_id + "-" + model_id;
}

std::string perf_make_session_trace_id(const std::string & session_id) {
    return "session-" + session_id;
}
