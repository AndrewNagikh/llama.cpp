#include "trace_recorder.h"

#include "hidden_transport.h"
#include "runtime_state.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <sstream>

namespace {

static trace_recorder * g_recorder = nullptr;

static std::string json_str(const std::string & s) {
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

static void write_raw_bin(
        const dist_debug_config & cfg,
        const std::string & name,
        const float * data,
        int64_t n) {
    if (!cfg.dump_raw_bins || data == nullptr || n <= 0) {
        return;
    }
    std::error_code ec;
    std::filesystem::create_directories(cfg.trace_dir, ec);
    const std::string path = cfg.trace_dir + "/" + name + ".bin";
    FILE * f = fopen(path.c_str(), "wb");
    if (!f) {
        return;
    }
    fwrite(data, sizeof(float), static_cast<size_t>(n), f);
    fclose(f);
}

static float logits_entropy(const float * logits, const int32_t n_vocab) {
    if (logits == nullptr || n_vocab <= 0) {
        return 0.0f;
    }
    float max_l = logits[0];
    for (int32_t i = 1; i < n_vocab; ++i) {
        max_l = std::max(max_l, logits[i]);
    }
    double sum = 0.0;
    for (int32_t i = 0; i < n_vocab; ++i) {
        sum += std::exp(static_cast<double>(logits[i] - max_l));
    }
    if (sum <= 0.0) {
        return 0.0f;
    }
    double ent = 0.0;
    for (int32_t i = 0; i < n_vocab; ++i) {
        const double p = std::exp(static_cast<double>(logits[i] - max_l)) / sum;
        if (p > 1e-12) {
            ent -= p * std::log(p);
        }
    }
    return static_cast<float>(ent);
}

static int32_t logits_argmax(const float * logits, const int32_t n_vocab) {
    if (logits == nullptr || n_vocab <= 0) {
        return -1;
    }
    int32_t best_i = 0;
    float best_v   = logits[0];
    for (int32_t i = 1; i < n_vocab; ++i) {
        if (logits[i] > best_v) {
            best_v = logits[i];
            best_i = i;
        }
    }
    return best_i;
}

} // namespace

trace_recorder::trace_recorder() : trace_recorder("unknown") {}

trace_recorder::trace_recorder(const std::string & role) : role_(role) {
    const dist_debug_config & cfg = dist_debug_config_get();
    if (!cfg.enabled) {
        return;
    }
    std::error_code ec;
    std::filesystem::create_directories(cfg.trace_dir, ec);
    const std::string sid = cfg.session_id.empty() ? "nosession" : cfg.session_id;
    const std::string nid = cfg.node_id.empty() ? "nonode" : cfg.node_id;
    trace_path_ = cfg.trace_dir + "/" + sid + "_" + nid + "_" + role_ + ".jsonl";
}

void trace_recorder::ensure_open() {
    if (!dist_debug_enabled()) {
        return;
    }
    if (out_.is_open()) {
        return;
    }
    out_.open(trace_path_, std::ios::app);
}

void trace_recorder::write_line(const std::string & json_line) {
    if (!dist_debug_enabled()) {
        return;
    }
    std::lock_guard<std::mutex> lock(mu_);
    ensure_open();
    if (!out_) {
        return;
    }
    out_ << json_line << '\n';
    out_.flush();
}

std::string trace_recorder::base_fields(
        const char * event,
        const int32_t step,
        const std::string & phase) const {
    const dist_debug_config & cfg = dist_debug_config_get();
    std::ostringstream os;
    os << "{"
       << "\"event\":" << json_str(event) << ","
       << "\"step\":" << step << ","
       << "\"phase\":" << json_str(phase) << ","
       << "\"worker\":" << json_str(role_) << ","
       << "\"session\":" << json_str(cfg.session_id) << ","
       << "\"node\":" << json_str(cfg.node_id) << ","
       << "\"ts_ms\":" << dist_debug_now_ms();
    return os.str();
}

void trace_recorder::set_step(const int32_t step) { cur_step_ = step; }
void trace_recorder::set_phase(const std::string & phase) { cur_phase_ = phase; }

void trace_recorder::emit_step_begin(
        const int32_t step,
        const std::string & phase,
        const int32_t token_id,
        const int32_t position,
        const int32_t seq_id) {
    cur_step_   = step;
    cur_phase_  = phase;
    std::ostringstream os;
    os << base_fields("step_begin", step, phase) << ","
       << "\"token\":" << token_id << ","
       << "\"position\":" << position << ","
       << "\"seq_id\":" << seq_id << "}";
    write_line(os.str());
}

void trace_recorder::emit_hidden(
        const int32_t step,
        const std::string & phase,
        const float * data,
        const int32_t n_tokens,
        const int32_t n_embd,
        const char * dump_tag) {
    if (data == nullptr || n_tokens <= 0 || n_embd <= 0) {
        return;
    }
    const int64_t n = static_cast<int64_t>(n_tokens) * n_embd;
    const tensor_stats stats = compute_tensor_stats(data, n);
    const dist_debug_config & cfg = dist_debug_config_get();
    if (dump_tag != nullptr) {
        char name[256];
        snprintf(name, sizeof(name), "hidden_%s_step_%03d_%s", dump_tag, step, role_.c_str());
        write_raw_bin(cfg, name, data, n);
    }
    std::ostringstream os;
    os << base_fields("hidden", step, phase) << ","
       << "\"n_tokens\":" << n_tokens << ","
       << "\"n_embd\":" << n_embd << ","
       << "\"hidden_shape\":[" << n_tokens << "," << n_embd << "],"
       << "\"dtype\":\"f32\","
       << "\"stats\":" << tensor_stats_json(stats) << "}";
    write_line(os.str());
}

void trace_recorder::emit_logits(
        const int32_t step,
        const std::string & phase,
        const float * logits,
        const int32_t vocab_size,
        const bool skip_sampler) {
    if (logits == nullptr || vocab_size <= 0) {
        return;
    }
    const tensor_stats stats = compute_tensor_stats(logits, vocab_size);
    const int32_t argmax     = logits_argmax(logits, vocab_size);
    const float argmax_score = argmax >= 0 ? logits[argmax] : 0.0f;
    const float entropy      = logits_entropy(logits, vocab_size);

    struct top_item { int32_t id; float logit; };
    std::vector<top_item> items(static_cast<size_t>(vocab_size));
    for (int32_t i = 0; i < vocab_size; ++i) {
        items[(size_t) i] = { i, logits[i] };
    }
    const size_t top_k = std::min<size_t>(20, items.size());
    std::partial_sort(
            items.begin(),
            items.begin() + static_cast<std::ptrdiff_t>(top_k),
            items.end(),
            [](const top_item & a, const top_item & b) { return a.logit > b.logit; });

    std::ostringstream os;
    os << base_fields("logits", step, phase) << ","
       << "\"vocab_size\":" << vocab_size << ","
       << "\"stats\":" << tensor_stats_json(stats) << ","
       << "\"argmax\":" << argmax << ","
       << "\"argmax_score\":" << argmax_score << ","
       << "\"entropy\":" << entropy << ","
       << "\"skip_sampler\":" << (skip_sampler ? "true" : "false") << ","
       << "\"top20_tokens\":[";
    for (size_t i = 0; i < top_k; ++i) {
        if (i) {
            os << ',';
        }
        os << items[i].id;
    }
    os << "],\"top20_logits\":[";
    for (size_t i = 0; i < top_k; ++i) {
        if (i) {
            os << ',';
        }
        os << items[i].logit;
    }
    os << "]}";
    write_line(os.str());

    const dist_debug_config & cfg = dist_debug_config_get();
    if (cfg.dump_raw_bins) {
        char name[256];
        snprintf(name, sizeof(name), "logits_step_%03d_%s", step, role_.c_str());
        write_raw_bin(cfg, name, logits, vocab_size);
    }
}

void trace_recorder::emit_position(
        const int32_t step,
        const std::string & phase,
        const int32_t position,
        const int32_t past_tokens,
        const int32_t seq_len,
        const int32_t batch_size) {
    std::ostringstream os;
    os << base_fields("position", step, phase) << ","
       << "\"position\":" << position << ","
       << "\"past_tokens\":" << past_tokens << ","
       << "\"seq_len\":" << seq_len << ","
       << "\"batch_size\":" << batch_size << "}";
    write_line(os.str());
}

void trace_recorder::emit_kv(
        const int32_t step,
        const std::string & phase,
        const int32_t kv_entries,
        const int32_t n_layer,
        const int32_t seq_len) {
    std::ostringstream os;
    os << base_fields("kv", step, phase) << ","
       << "\"kv_entries\":" << kv_entries << ","
       << "\"layer_count\":" << n_layer << ","
       << "\"seq_len\":" << seq_len << "}";
    write_line(os.str());
}

void trace_recorder::emit_token_selected(
        const int32_t step,
        const std::string & phase,
        const int32_t token_id,
        const int32_t position,
        const bool via_argmax) {
    std::ostringstream os;
    os << base_fields("token_selected", step, phase) << ","
       << "\"token\":" << token_id << ","
       << "\"position\":" << position << ","
       << "\"via_argmax\":" << (via_argmax ? "true" : "false") << "}";
    write_line(os.str());
}

void trace_recorder::emit_runtime_state(const runtime_state_snapshot & snap) {
    std::ostringstream os;
    os << base_fields("runtime_state", snap.step, snap.phase) << ","
       << "\"state\":" << runtime_state_json(snap);
    os << "}";
    write_line(os.str());
}

void trace_recorder::emit_transport(const hidden_transport_trace & tr) {
    std::ostringstream os;
    os << base_fields("transport", tr.step, tr.phase) << ","
       << "\"transport\":" << hidden_transport_trace_json(tr) << "}";
    write_line(os.str());
}

trace_recorder * dist_debug_recorder() {
    if (!dist_debug_enabled()) {
        return nullptr;
    }
    if (g_recorder == nullptr) {
        const dist_debug_config & cfg = dist_debug_config_get();
        g_recorder = new trace_recorder(cfg.worker_id.empty() ? "worker" : cfg.worker_id);
    }
    return g_recorder;
}

void dist_debug_reset_recorder(const std::string & role) {
    if (g_recorder != nullptr) {
        delete g_recorder;
        g_recorder = nullptr;
    }
    if (dist_debug_enabled()) {
        g_recorder = new trace_recorder(role);
    }
}
