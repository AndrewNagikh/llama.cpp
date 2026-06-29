#include "context_params_verify.h"

#include "../../../src/llama-context.h"

#include "ggml-backend.h"

#include <sstream>

static std::string to_string_bool(const bool v) {
    return v ? "true" : "false";
}

static void json_escape(std::ostringstream & os, const std::string & s) {
    os << '"';
    for (char c : s) {
        if (c == '"') {
            os << "\\\"";
        } else if (c == '\\') {
            os << "\\\\";
        } else {
            os << c;
        }
    }
    os << '"';
}

context_params_snapshot capture_context_params(
        llama_context * ctx,
        const char * label) {
    context_params_snapshot snap{};
    snap.label = label ? label : "";

    if (!ctx) {
        return snap;
    }

    const llama_cparams & cp = ctx->get_cparams();
    const llama_model * model = llama_get_model(ctx);

    snap.n_ctx            = static_cast<int32_t>(cp.n_ctx);
    snap.n_batch          = static_cast<int32_t>(cp.n_batch);
    snap.n_ubatch         = static_cast<int32_t>(cp.n_ubatch);
    snap.causal_attn      = cp.causal_attn;
    snap.flash_attn       = cp.flash_attn;
    snap.auto_fa          = cp.auto_fa;
    snap.rope_freq_base   = cp.rope_freq_base;
    snap.rope_freq_scale  = cp.rope_freq_scale;
    snap.n_ctx_orig_yarn  = cp.n_ctx_orig_yarn;
    snap.yarn_ext_factor  = cp.yarn_ext_factor;
    snap.yarn_attn_factor = cp.yarn_attn_factor;
    snap.yarn_beta_fast   = cp.yarn_beta_fast;
    snap.yarn_beta_slow   = cp.yarn_beta_slow;
    snap.embeddings       = cp.embeddings;
    snap.pooling_type     = static_cast<int32_t>(cp.pooling_type);
    snap.ctx_type         = static_cast<int32_t>(cp.ctx_type);
    snap.offload_kqv      = cp.offload_kqv;
    snap.kv_unified       = cp.kv_unified;
    snap.layer_start      = cp.layer_start;
    snap.layer_end        = cp.layer_end;
    snap.n_threads        = cp.n_threads;
    snap.n_threads_batch  = cp.n_threads_batch;
    snap.no_perf          = cp.no_perf;
    snap.pipeline_parallel = cp.pipeline_parallel;
    snap.kv_n_layer       = llama_model_n_layer(model);

    llama_memory_t mem = llama_get_memory(ctx);
    if (mem) {
        const llama_pos seq_max = llama_memory_seq_pos_max(mem, 0);
        snap.kv_entries         = seq_max >= 0 ? static_cast<int32_t>(seq_max + 1) : 0;
    }

    ggml_backend_t backend = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
    if (backend) {
        snap.backend = ggml_backend_name(backend);
        ggml_backend_free(backend);
    }

    return snap;
}

static void add_diff(
        context_params_diff_report & report,
        const char * field,
        const std::string & mono,
        const std::string & worker) {
    if (mono == worker) {
        return;
    }
    report.diffs.push_back({ field, mono, worker });
}

context_params_diff_report compare_context_params(
        llama_context * mono_ctx,
        llama_context * worker_ctx) {
    context_params_diff_report report{};
    report.mono   = capture_context_params(mono_ctx, "mono");
    report.worker = capture_context_params(worker_ctx, "worker_final");

    add_diff(report, "n_ctx", std::to_string(report.mono.n_ctx), std::to_string(report.worker.n_ctx));
    add_diff(report, "n_batch", std::to_string(report.mono.n_batch), std::to_string(report.worker.n_batch));
    add_diff(report, "n_ubatch", std::to_string(report.mono.n_ubatch), std::to_string(report.worker.n_ubatch));
    add_diff(report, "causal", to_string_bool(report.mono.causal_attn), to_string_bool(report.worker.causal_attn));
    add_diff(report, "flash_attn", to_string_bool(report.mono.flash_attn), to_string_bool(report.worker.flash_attn));
    add_diff(report, "auto_fa", to_string_bool(report.mono.auto_fa), to_string_bool(report.worker.auto_fa));
    add_diff(report, "rope_freq_base", std::to_string(report.mono.rope_freq_base), std::to_string(report.worker.rope_freq_base));
    add_diff(report, "rope_freq_scale", std::to_string(report.mono.rope_freq_scale), std::to_string(report.worker.rope_freq_scale));
    add_diff(report, "yarn_ext_factor", std::to_string(report.mono.yarn_ext_factor), std::to_string(report.worker.yarn_ext_factor));
    add_diff(report, "embeddings", to_string_bool(report.mono.embeddings), to_string_bool(report.worker.embeddings));
    add_diff(report, "pooling", std::to_string(report.mono.pooling_type), std::to_string(report.worker.pooling_type));
    add_diff(report, "offload_kqv", to_string_bool(report.mono.offload_kqv), to_string_bool(report.worker.offload_kqv));
    add_diff(report, "kv_unified", to_string_bool(report.mono.kv_unified), to_string_bool(report.worker.kv_unified));
    add_diff(report, "no_perf", to_string_bool(report.mono.no_perf), to_string_bool(report.worker.no_perf));
    add_diff(report, "pipeline_parallel", to_string_bool(report.mono.pipeline_parallel), to_string_bool(report.worker.pipeline_parallel));
    add_diff(report, "n_threads", std::to_string(report.mono.n_threads), std::to_string(report.worker.n_threads));
    add_diff(report, "n_threads_batch", std::to_string(report.mono.n_threads_batch), std::to_string(report.worker.n_threads_batch));

    report.match   = report.diffs.empty();
    report.message = report.match ? "context params match" : "context params differ count=" +
            std::to_string(report.diffs.size());
    return report;
}

std::string context_params_snapshot_json(const context_params_snapshot & snap) {
    std::ostringstream os;
    os << "{"
       << "\"label\":";
    json_escape(os, snap.label);
    os << ",\"n_ctx\":" << snap.n_ctx
       << ",\"n_batch\":" << snap.n_batch
       << ",\"n_ubatch\":" << snap.n_ubatch
       << ",\"causal\":" << (snap.causal_attn ? "true" : "false")
       << ",\"flash_attn\":" << (snap.flash_attn ? "true" : "false")
       << ",\"auto_fa\":" << (snap.auto_fa ? "true" : "false")
       << ",\"rope_freq_base\":" << snap.rope_freq_base
       << ",\"rope_freq_scale\":" << snap.rope_freq_scale
       << ",\"n_ctx_orig_yarn\":" << snap.n_ctx_orig_yarn
       << ",\"yarn_ext_factor\":" << snap.yarn_ext_factor
       << ",\"yarn_attn_factor\":" << snap.yarn_attn_factor
       << ",\"yarn_beta_fast\":" << snap.yarn_beta_fast
       << ",\"yarn_beta_slow\":" << snap.yarn_beta_slow
       << ",\"embeddings\":" << (snap.embeddings ? "true" : "false")
       << ",\"pooling\":" << snap.pooling_type
       << ",\"ctx_type\":" << snap.ctx_type
       << ",\"offload_kqv\":" << (snap.offload_kqv ? "true" : "false")
       << ",\"kv_unified\":" << (snap.kv_unified ? "true" : "false")
       << ",\"kv_entries\":" << snap.kv_entries
       << ",\"kv_n_layer\":" << snap.kv_n_layer
       << ",\"layer_start\":" << snap.layer_start
       << ",\"layer_end\":" << snap.layer_end
       << ",\"n_threads\":" << snap.n_threads
       << ",\"n_threads_batch\":" << snap.n_threads_batch
       << ",\"no_perf\":" << (snap.no_perf ? "true" : "false")
       << ",\"pipeline_parallel\":" << (snap.pipeline_parallel ? "true" : "false")
       << ",\"backend\":";
    json_escape(os, snap.backend);
    os << "}";
    return os.str();
}

std::string context_params_diff_json(const context_params_diff_report & report) {
    std::ostringstream os;
    os << "{"
       << "\"match\":" << (report.match ? "true" : "false")
       << ",\"message\":";
    json_escape(os, report.message);
    os << ",\"mono\":" << context_params_snapshot_json(report.mono)
       << ",\"worker\":" << context_params_snapshot_json(report.worker)
       << ",\"diffs\":[";
    for (size_t i = 0; i < report.diffs.size(); ++i) {
        if (i > 0) {
            os << ',';
        }
        const auto & d = report.diffs[i];
        os << "{\"field\":";
        json_escape(os, d.field);
        os << ",\"mono\":";
        json_escape(os, d.mono_value);
        os << ",\"worker\":";
        json_escape(os, d.worker_value);
        os << "}";
    }
    os << "]}";
    return os.str();
}
