#include "layer_equivalence.h"

#include "graph_equivalence.h"

#include <sstream>

static layer_equivalence_diagnostics diagnose_layer0_failure(
        const layer_boundary_sample & mono,
        const layer_boundary_sample & worker) {
    layer_equivalence_diagnostics diag{};
    diag.fail_layer = 0;
    diag.fail_kind  = layer_fail_kind::output;

    if (mono.has_input && worker.has_input) {
        diag.embedding_parity        = compare_tensors(mono.input.data(), worker.input.data(), mono.n_embd);
        diag.embedding_input_match   = diag.embedding_parity.match;
    }

    diag.positions_match = (mono.positions == worker.positions);
    diag.kv_seq_match    = (mono.kv_seq_max == worker.kv_seq_max);

    if (!diag.embedding_input_match && mono.has_input) {
        diag.hint = "layer0 input (embedding output) differs — check embedding weights / token ids";
    } else if (!diag.positions_match) {
        diag.hint = "position ids differ — check RoPE / batch positions";
    } else if (!diag.kv_seq_match) {
        diag.hint = "KV sequence length differs after layer0 — check attention mask / cache";
    } else {
        diag.hint = "layer0 output differs with matching embedding input — check attention / RoPE / weights";
    }

    return diag;
}

static layer_equivalence_diagnostics diagnose_layerN_failure(
        const int32_t layer,
        const layer_boundary_sample & mono,
        const layer_boundary_sample & worker) {
    layer_equivalence_diagnostics diag{};
    diag.fail_layer = layer;
    diag.fail_kind  = layer_fail_kind::output;

    if (mono.has_input && worker.has_input) {
        diag.embedding_parity      = compare_tensors(mono.input.data(), worker.input.data(), mono.n_embd);
        diag.embedding_input_match = diag.embedding_parity.match;
    }

    diag.positions_match = (mono.positions == worker.positions);
    diag.kv_seq_match    = (mono.kv_seq_max == worker.kv_seq_max);

    if (!diag.embedding_input_match) {
        diag.hint = "layer input differs — upstream layer or residual path diverged";
    } else if (!diag.kv_seq_match) {
        diag.hint = "KV cache differs — check graph execution / layer state";
    } else {
        diag.hint = "layer input matches but output differs — check layer weights / FFN / attention";
    }

    return diag;
}

static void maybe_trace_boundary(
        const layer_trace_config & cfg,
        const std::string & label,
        const layer_boundary_sample & sample,
        const parity_metrics * parity) {
    if (!cfg.enabled) {
        return;
    }

    layer_boundary_dump dump{};
    dump.layer_index  = sample.layer_index;
    dump.token_index  = sample.token_index;
    dump.n_embd       = sample.n_embd;
    dump.input        = sample.input_stats;
    dump.output       = sample.output_stats;
    dump.positions    = sample.positions;
    dump.kv_seq_max   = sample.kv_seq_max;
    dump.input_ok     = sample.has_input;
    dump.output_ok    = sample.has_output;
    if (parity) {
        dump.parity = *parity;
    }
    layer_trace_write_boundary(cfg, label, dump);
}

layer_equivalence_report verify_layer_equivalence(
        const std::string & mono_path,
        const std::string & worker_path,
        const std::string & prompt,
        int32_t max_layer,
        const layer_trace_config * trace_cfg_in) {
    layer_equivalence_report report{};
    const layer_trace_config cfg = trace_cfg_in ? *trace_cfg_in : layer_trace_load_config();

    layer_runner_ctx mono_rt   = layer_runner_load(mono_path);
    layer_runner_ctx worker_rt = layer_runner_load(worker_path);

    if (!mono_rt.model || !worker_rt.model) {
        report.message = "model load failed (mono=" + std::to_string(mono_rt.model != nullptr) +
                         " worker=" + std::to_string(worker_rt.model != nullptr) + ")";
        layer_runner_free(mono_rt);
        layer_runner_free(worker_rt);
        return report;
    }

    const std::vector<llama_token> tokens = layer_runner_tokenize(mono_rt, prompt);
    if (tokens.empty()) {
        report.message = "tokenization failed";
        layer_runner_free(mono_rt);
        layer_runner_free(worker_rt);
        return report;
    }

    const int32_t mono_layers   = llama_model_n_layer(mono_rt.model);
    const int32_t worker_layers = llama_model_n_layer(worker_rt.model);
    int32_t limit               = max_layer < 0 ? std::min(mono_layers, worker_layers) : max_layer;
    limit                       = std::min(limit, std::min(mono_layers, worker_layers));

    const bool capture_input = cfg.capture_input;

    for (int32_t lid = 0; lid < limit; ++lid) {
        layer_parity_row row{};
        row.layer_index = lid;

        const layer_boundary_sample mono =
                layer_runner_capture_boundary(mono_rt, tokens, lid, capture_input);
        const layer_boundary_sample worker =
                layer_runner_capture_boundary(worker_rt, tokens, lid, capture_input);

        if (!mono.has_output || !worker.has_output) {
            row.pass    = false;
            row.message = "missing layer output activation";
            report.rows.push_back(row);
            report.first_fail_layer = lid;
            report.message          = "Layer " + std::to_string(lid) + " FAIL (no output)";
            break;
        }

        row.output_parity = compare_tensors(mono.output.data(), worker.output.data(), mono.n_embd);
        if (mono.has_input && worker.has_input) {
            row.input_parity = compare_tensors(mono.input.data(), worker.input.data(), mono.n_embd);
        }

        row.pass = row.output_parity.match;
        if (row.pass) {
            row.message = "PASS";
            fprintf(stdout, "Layer %d PASS\n", lid);
        } else {
            row.message = "output mismatch sha256 mono=" + mono.output_stats.sha256.substr(0, 12) +
                          " worker=" + worker.output_stats.sha256.substr(0, 12);
            fprintf(stdout, "Layer %d FAIL\n", lid);
            report.first_fail_layer = lid;
            report.message          = "Layer " + std::to_string(lid) + " FAIL";

            if (lid == 0) {
                report.diagnostics = diagnose_layer0_failure(mono, worker);
            } else {
                report.diagnostics = diagnose_layerN_failure(lid, mono, worker);
            }

            if (cfg.enabled || cfg.dump_graph) {
                const auto graph_cmp = compare_layer_graphs(
                        mono_rt.ctx,
                        worker_rt.ctx,
                        0,
                        lid + 1,
                        (uint32_t) tokens.size());
                layer_trace_write_graph(
                        cfg,
                        "mono",
                        0,
                        lid + 1,
                        graph_summary_json(graph_cmp.mono));
                layer_trace_write_graph(
                        cfg,
                        "worker",
                        0,
                        lid + 1,
                        graph_summary_json(graph_cmp.worker));
                if (!graph_cmp.match) {
                    report.diagnostics.hint += " | graph: " + graph_cmp.message;
                }
            }

            if (cfg.trace_block_ops &&
                (cfg.trace_block_layer < 0 || cfg.trace_block_layer == lid)) {
                const auto block_mono = graph_summarize_layer(
                        mono_rt.ctx, lid, lid + 1, (uint32_t) tokens.size());
                const auto block_worker = graph_summarize_layer(
                        worker_rt.ctx, lid, lid + 1, (uint32_t) tokens.size());
                layer_trace_write_graph(
                        cfg,
                        "mono_block",
                        lid,
                        lid + 1,
                        graph_summary_json(block_mono));
                layer_trace_write_graph(
                        cfg,
                        "worker_block",
                        lid,
                        lid + 1,
                        graph_summary_json(block_worker));
            }

            maybe_trace_boundary(cfg, "mono", mono, &row.output_parity);
            maybe_trace_boundary(cfg, "worker", worker, &row.output_parity);
            report.rows.push_back(row);
            break;
        }

        maybe_trace_boundary(cfg, "mono", mono, &row.output_parity);
        maybe_trace_boundary(cfg, "worker", worker, &row.output_parity);
        report.rows.push_back(row);
    }

    report.layers_checked = (int32_t) report.rows.size();
    report.all_pass       = report.first_fail_layer < 0 && !report.rows.empty();
    if (report.all_pass) {
        report.message = "all " + std::to_string(report.layers_checked) + " layers PASS";
    }

    layer_runner_free(mono_rt);
    layer_runner_free(worker_rt);
    return report;
}

std::string layer_equivalence_report_json(const layer_equivalence_report & report) {
    std::ostringstream os;
    os << "{"
       << "\"all_pass\":" << (report.all_pass ? "true" : "false") << ","
       << "\"layers_checked\":" << report.layers_checked << ","
       << "\"first_fail_layer\":" << report.first_fail_layer << ","
       << "\"message\":\"" << report.message << "\","
       << "\"diagnostics\":{"
       << "\"fail_layer\":" << report.diagnostics.fail_layer << ","
       << "\"hint\":\"" << report.diagnostics.hint << "\","
       << "\"embedding_input_match\":" << (report.diagnostics.embedding_input_match ? "true" : "false") << ","
       << "\"positions_match\":" << (report.diagnostics.positions_match ? "true" : "false") << ","
       << "\"kv_seq_match\":" << (report.diagnostics.kv_seq_match ? "true" : "false")
       << "},"
       << "\"rows\":[";
    for (size_t i = 0; i < report.rows.size(); ++i) {
        if (i > 0) {
            os << ',';
        }
        const auto & row = report.rows[i];
        os << "{"
           << "\"layer\":" << row.layer_index << ","
           << "\"pass\":" << (row.pass ? "true" : "false") << ","
           << "\"message\":\"" << row.message << "\","
           << "\"output_parity\":" << parity_metrics_json(row.output_parity) << ","
           << "\"input_parity\":" << parity_metrics_json(row.input_parity)
           << "}";
    }
    os << "]}";
    return os.str();
}
