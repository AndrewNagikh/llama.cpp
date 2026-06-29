#include "decode_loop_parity.h"

#include "nlohmann/json.hpp"

#include <cmath>
#include <fstream>

using json = nlohmann::json;

static tensor_stats parse_stats(const json & j) {
    tensor_stats s{};
    if (!j.is_object()) {
        return s;
    }
    s.n_elements = j.value("n_elements", 0);
    s.min_val    = j.value("min", 0.0f);
    s.max_val    = j.value("max", 0.0f);
    s.mean       = j.value("mean", 0.0f);
    s.stddev     = j.value("std", 0.0f);
    s.l2_norm    = j.value("l2_norm", 0.0);
    s.sha256     = j.value("sha256", std::string{});
    s.has_nan    = j.value("has_nan", false);
    s.has_inf    = j.value("has_inf", false);
    return s;
}

parsed_trace parse_trace_jsonl(const std::string & path) {
    parsed_trace trace;
    trace.path = path;
    std::ifstream in(path);
    if (!in) {
        return trace;
    }
    std::string line;
    while (std::getline(in, line)) {
        if (line.empty()) {
            continue;
        }
        json j;
        try {
            j = json::parse(line);
        } catch (...) {
            continue;
        }
        trace_event ev;
        ev.event    = j.value("event", "");
        ev.step     = j.value("step", -1);
        ev.phase    = j.value("phase", "");
        ev.worker   = j.value("worker", "");
        ev.session  = j.value("session", "");
        ev.node     = j.value("node", "");
        ev.token    = j.value("token", -1);
        ev.position = j.value("position", -1);
        ev.seq_id   = j.value("seq_id", 0);
        ev.n_tokens = j.value("n_tokens", 0);
        ev.n_embd   = j.value("n_embd", 0);
        ev.vocab_size = j.value("vocab_size", 0);
        ev.argmax   = j.value("argmax", -1);
        ev.argmax_score = j.value("argmax_score", 0.0f);
        ev.entropy  = j.value("entropy", 0.0f);
        ev.kv_entries = j.value("kv_entries", 0);
        ev.seq_len  = j.value("seq_len", 0);
        ev.past_tokens = j.value("past_tokens", 0);
        if (j.contains("stats") && ev.event == "hidden") {
            ev.hidden_stats = parse_stats(j["stats"]);
            ev.has_hidden   = true;
            ev.sha256       = ev.hidden_stats.sha256;
        }
        if (j.contains("stats") && ev.event == "logits") {
            ev.logits_stats = parse_stats(j["stats"]);
            ev.has_logits   = true;
            ev.sha256       = ev.logits_stats.sha256;
        }
        if (trace.worker.empty() && !ev.worker.empty()) {
            trace.worker = ev.worker;
        }
        trace.events.push_back(std::move(ev));
    }
    return trace;
}

static const trace_event * find_event(
        const parsed_trace & trace,
        const char * event_name,
        int32_t step,
        const std::string & phase) {
    for (const auto & ev : trace.events) {
        if (ev.event == event_name && ev.step == step && ev.phase == phase) {
            return &ev;
        }
    }
    return nullptr;
}

std::string divergence_kind_name(const divergence_kind kind) {
    switch (kind) {
        case divergence_kind::token:           return "token";
        case divergence_kind::position:        return "position";
        case divergence_kind::hidden:          return "hidden";
        case divergence_kind::logits:          return "logits";
        case divergence_kind::kv:              return "kv";
        case divergence_kind::selected_token:  return "selected_token";
        case divergence_kind::missing_event:   return "missing_event";
        default:                               return "none";
    }
}

static divergence_report make_report(
        divergence_kind kind,
        int32_t step,
        const std::string & phase,
        const std::string & worker,
        const std::string & mono_src,
        const std::string & dist_src,
        const std::string & field,
        const std::string & message) {
    divergence_report r;
    r.kind         = kind;
    r.step         = step;
    r.phase        = phase;
    r.worker       = worker;
    r.mono_source  = mono_src;
    r.dist_source  = dist_src;
    r.field        = field;
    r.message      = message;
    switch (kind) {
        case divergence_kind::position:
            r.root_cause_hint = "Root Cause: Position mismatch";
            break;
        case divergence_kind::hidden:
            r.root_cause_hint = "Root Cause: Hidden divergence after worker " + worker;
            break;
        case divergence_kind::logits:
            r.root_cause_hint = "Root Cause: Final logits divergence";
            break;
        case divergence_kind::kv:
            r.root_cause_hint = "Root Cause: KV cache growth mismatch";
            break;
        case divergence_kind::selected_token:
            r.root_cause_hint = "Root Cause: Token selection diverged (check sampler vs logits)";
            break;
        default:
            r.root_cause_hint = "Root Cause: " + field + " mismatch at step " + std::to_string(step);
            break;
    }
    return r;
}

divergence_report find_first_divergence(
        const parsed_trace & mono,
        const parsed_trace & dist_entry,
        const parsed_trace & dist_final,
        const parsed_trace & dist_middle) {
    divergence_report ok{};
    ok.kind = divergence_kind::none;

    const parsed_trace * dist_workers[] = { &dist_entry, &dist_middle, &dist_final };

    for (const auto & ev : mono.events) {
        if (ev.event != "step_begin" && ev.event != "token_selected" &&
                ev.event != "position" && ev.event != "kv" &&
                ev.event != "hidden" && ev.event != "logits") {
            continue;
        }

        if (ev.event == "hidden" || ev.event == "logits" || ev.event == "token_selected") {
            const parsed_trace * dist = nullptr;
            if (ev.event == "logits" || (ev.event == "token_selected" && ev.phase != "prefill")) {
                dist = &dist_final;
            } else if (ev.event == "hidden" && ev.phase == "prefill") {
                dist = &dist_entry;
            } else if (ev.event == "token_selected") {
                dist = &dist_final;
            } else {
                dist = &dist_entry;
            }

            const trace_event * dist_ev = find_event(*dist, ev.event.c_str(), ev.step, ev.phase);
            if (!dist_ev) {
                return make_report(
                        divergence_kind::missing_event,
                        ev.step, ev.phase, dist->worker,
                        mono.path, dist->path, ev.event,
                        "distributed trace missing " + ev.event);
            }

            if (ev.event == "hidden" && ev.has_hidden && dist_ev->has_hidden) {
                if (ev.hidden_stats.sha256 != dist_ev->hidden_stats.sha256) {
                    auto r = make_report(
                            divergence_kind::hidden,
                            ev.step, ev.phase, dist->worker,
                            mono.path, dist->path, "hidden.sha256",
                            "hidden sha256 differs mono=" + ev.hidden_stats.sha256.substr(0, 12) +
                                    " dist=" + dist_ev->hidden_stats.sha256.substr(0, 12));
                    r.metrics.max_abs_err = std::fabs(ev.hidden_stats.mean - dist_ev->hidden_stats.mean);
                    return r;
                }
            }

            if (ev.event == "logits" && ev.has_logits && dist_ev->has_logits) {
                if (ev.logits_stats.sha256 != dist_ev->logits_stats.sha256) {
                    auto r = make_report(
                            divergence_kind::logits,
                            ev.step, ev.phase, dist->worker,
                            mono.path, dist->path, "logits.sha256",
                            "logits sha256 differs argmax mono=" + std::to_string(ev.argmax) +
                                    " dist=" + std::to_string(dist_ev->argmax));
                    return r;
                }
            }

            if (ev.event == "token_selected" && ev.token != dist_ev->token) {
                return make_report(
                        divergence_kind::selected_token,
                        ev.step, ev.phase, dist->worker,
                        mono.path, dist->path, "token",
                        "token mono=" + std::to_string(ev.token) +
                                " dist=" + std::to_string(dist_ev->token));
            }
        }

        if (ev.event == "position") {
            for (const parsed_trace * dist : dist_workers) {
                if (dist->events.empty()) {
                    continue;
                }
                const trace_event * dist_ev = find_event(*dist, "position", ev.step, ev.phase);
                if (!dist_ev) {
                    continue;
                }
                if (ev.position != dist_ev->position) {
                    return make_report(
                            divergence_kind::position,
                            ev.step, ev.phase, dist->worker,
                            mono.path, dist->path, "position",
                            "position mono=" + std::to_string(ev.position) +
                                    " dist=" + std::to_string(dist_ev->position));
                }
            }
        }

        if (ev.event == "kv") {
            for (const parsed_trace * dist : dist_workers) {
                if (dist->events.empty()) {
                    continue;
                }
                const trace_event * dist_ev = find_event(*dist, "kv", ev.step, ev.phase);
                if (!dist_ev) {
                    continue;
                }
                if (ev.kv_entries != dist_ev->kv_entries) {
                    return make_report(
                            divergence_kind::kv,
                            ev.step, ev.phase, dist->worker,
                            mono.path, dist->path, "kv_entries",
                            "kv mono=" + std::to_string(ev.kv_entries) +
                                    " dist=" + std::to_string(dist_ev->kv_entries));
                }
            }
        }
    }

    return ok;
}
