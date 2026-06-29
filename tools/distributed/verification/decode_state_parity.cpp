#include "decode_state_parity.h"

#include <map>

static decode_state_row row_from_event(const trace_event & ev) {
    decode_state_row row{};
    row.step           = ev.step;
    row.phase          = ev.phase;
    row.worker         = ev.worker;
    row.token          = ev.token;
    row.position       = ev.position;
    row.n_past         = ev.past_tokens;
    row.kv_entries     = ev.kv_entries;
    row.hidden_sha256  = ev.hidden_stats.sha256;
    row.has_runtime_state = ev.has_hidden || ev.event == "runtime_state";
    return row;
}

static const trace_event * find_event(
        const parsed_trace & trace,
        const char * event_name,
        int32_t step,
        const char * phase) {
    for (const auto & ev : trace.events) {
        if (ev.event == event_name && ev.step == step &&
                (phase == nullptr || ev.phase == phase)) {
            return &ev;
        }
    }
    return nullptr;
}

decode_state_parity_report compare_decode_state_traces(
        const parsed_trace & mono,
        const parsed_trace & entry,
        const parsed_trace & middle,
        const parsed_trace & final_trace) {
    decode_state_parity_report report{};

    std::map<int32_t, decode_state_row> mono_by_step;
    for (const auto & ev : mono.events) {
        if (ev.event == "hidden" || ev.event == "runtime_state" || ev.event == "token_selected") {
            mono_by_step[ev.step] = row_from_event(ev);
        }
    }

    auto check_worker = [&](const parsed_trace & worker, const char * name) {
        for (const auto & ev : worker.events) {
            if (ev.event != "hidden" && ev.event != "runtime_state" &&
                ev.event != "token_selected" && ev.event != "transport") {
                continue;
            }

            decode_state_row row = row_from_event(ev);
            report.rows.push_back(row);

            const auto it = mono_by_step.find(ev.step);
            if (it == mono_by_step.end()) {
                continue;
            }

            if (ev.event == "token_selected" && ev.token >= 0 && it->second.token >= 0 &&
                    ev.token != it->second.token) {
                report.all_pass          = false;
                report.first_fail_step   = ev.step;
                report.first_fail_worker = name;
                report.field             = "token";
                report.message           = std::string(name) + " token mismatch at step " +
                        std::to_string(ev.step);
                return false;
            }

            if (ev.event == "hidden" && ev.has_hidden && !it->second.hidden_sha256.empty() &&
                    ev.hidden_stats.sha256 != it->second.hidden_sha256) {
                report.all_pass          = false;
                report.first_fail_step   = ev.step;
                report.first_fail_worker = name;
                report.field             = "hidden_sha256";
                report.message           = std::string(name) + " hidden differs at step " +
                        std::to_string(ev.step);
                return false;
            }

            if (ev.event == "runtime_state" || ev.event == "position") {
                if (ev.kv_entries != it->second.kv_entries && it->second.kv_entries > 0) {
                    report.all_pass          = false;
                    report.first_fail_step   = ev.step;
                    report.first_fail_worker = name;
                    report.field             = "kv_entries";
                    report.message           = std::string(name) + " kv mismatch at step " +
                            std::to_string(ev.step);
                    return false;
                }
            }
        }
        return true;
    };

    report.all_pass = true;
    if (!check_worker(entry, "entry")) {
        return report;
    }
    if (!middle.events.empty() && !check_worker(middle, "middle")) {
        return report;
    }
    if (!final_trace.events.empty() && !check_worker(final_trace, "final")) {
        return report;
    }

    report.message = "decode state parity PASS steps=" + std::to_string(report.rows.size());
    return report;
}

decode_state_parity_report compare_runtime_state_vs_mono(
        const std::string & mono_trace_path,
        const std::string & entry_trace_path,
        const std::string & middle_trace_path,
        const std::string & final_trace_path) {
    const parsed_trace mono   = parse_trace_jsonl(mono_trace_path);
    const parsed_trace entry  = parse_trace_jsonl(entry_trace_path);
    const parsed_trace middle = middle_trace_path.empty() ? parsed_trace{}
                                                          : parse_trace_jsonl(middle_trace_path);
    const parsed_trace finalt = final_trace_path.empty() ? parsed_trace{}
                                                         : parse_trace_jsonl(final_trace_path);
    return compare_decode_state_traces(mono, entry, middle, finalt);
}
