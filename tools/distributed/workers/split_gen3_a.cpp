// Process A: entry stage - layers [0, layer_end)

#include "ggml-backend.h"
#include "ggml.h"
#include "split_gen_model_load.h"
#include "split_gen_common.h"
#include "../runtime_debug/debug_hooks.h"
#include "../runtime_debug/runtime_debug.h"
#include "../runtime_debug/trace_recorder.h"
#include "../runtime_debug/hidden_transport.h"
#include "../transport/split_tcp_wire.h"
#include "../transport/runtime_protocol.h"
#include "../runtime_debug/hidden_transport_breakdown.h"
#include "../runtime_debug/perf_trace.h"
#include "../runtime_debug/perf_ggml.h"
#include "wave_inbound_queue.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <mutex>
#include <string>
#include <thread>
#include <utility>
#include <vector>

static void usage(const char * prog) {
    fprintf(stderr,
            "usage: %s MODEL --ctrl-port PORT --b-port PORT [--layer-end N] "
            "[--ready-file PATH]\n",
            prog);
}

static bool forward_to_peer(
        int peer_fd,
        llama_context * ctx,
        int32_t n_tokens,
        int32_t n_embd,
        int32_t layer_end,
        int32_t pos_start,
        bool include_logits,
        int32_t n_vocab,
        double ms_a,
        bool next_is_final,
        int32_t debug_step,
        const char * phase,
        trace_recorder * dbg,
        split_gen3_a_resp & resp,
        std::vector<float> & logits_out) {
    perf_trace_refresh_context();
    perf_trace_set_wave_id(perf_trace_wave_id_from_step(phase, debug_step));
    const int32_t tok_idx = (phase && std::strcmp(phase, "decode") == 0) ? debug_step : -1;

    std::vector<float> hidden_buf;
    hidden_pack_stats pack_stats{};
    if (perf_trace_enabled()) {
        if (phase && std::strcmp(phase, "decode") == 0) {
            perf_trace_ensure_decode_context(tok_idx, perf_trace_wave_id_from_step(phase, debug_step));
        } else {
            perf_trace_set_wave_id(perf_trace_wave_id_from_step(phase, debug_step));
        }
    }
    if (!hidden_pack_gather_stage_hidden(ctx, n_tokens, n_embd, hidden_buf, pack_stats.gather)) {
        fprintf(stderr, "gen3_a: missing hidden state\n");
        return false;
    }
    const float * hidden = hidden_buf.data();
    const int32_t payload_bytes = n_tokens * n_embd * (int32_t) sizeof(float);

    hidden_transport_trace tr_send = dist_debug_transport_send(
            debug_step, phase, "ab", n_tokens, n_embd, layer_end, pos_start, hidden, 0.0);
    if (dbg) {
        dbg->emit_transport(tr_send);
    }

    split_gen_pipe_trace("entry", phase, "send_hidden_enter", n_tokens, n_embd, pos_start, -1, layer_end);
    if (!hidden_pack_send_ab_hidden(peer_fd, n_tokens, n_embd, layer_end, pos_start,
            include_logits ? 1 : 0, hidden, pack_stats.send)) {
        fprintf(stderr, "gen3_a: send hidden failed\n");
        return false;
    }
    split_gen_pipe_trace("entry", phase, "send_hidden_exit", n_tokens, n_embd, pos_start, -1, layer_end);

    if (perf_trace_enabled()) {
        pack_stats.pack_total_us = pack_stats.gather.alloc_us + pack_stats.gather.sync_us
                + pack_stats.gather.gather_us
                + pack_stats.gather.copy_us + pack_stats.gather.serialize_us
                + pack_stats.send.frame_us + pack_stats.send.send_us;
        hidden_pack_emit_breakdown_spans(pack_stats, tok_idx, payload_bytes);
    }

    resp.magic          = SPLIT_GEN_MAGIC;
    resp.version        = SPLIT_GEN3_VERSION;
    resp.include_logits = include_logits ? 1 : 0;
    resp.n_vocab        = n_vocab;
    resp.ms_a_compute   = ms_a;
    resp.ms_ab_xfer     = pack_stats.send.send_us / 1000.0;
    logits_out.clear();

    if (next_is_final) {
        split_gen3_c_resp cresp{};
        if (!split_gen3_recv_c_resp(peer_fd, cresp)) {
            fprintf(stderr, "gen3_a: recv final resp failed\n");
            return false;
        }
        resp.token_id       = cresp.token_id;
        resp.ms_b_compute   = 0.0;
        resp.ms_bc_xfer     = 0.0;
        resp.ms_c_compute   = cresp.ms_compute;
        resp.ms_c_sample    = cresp.ms_sample;
        resp.accepted_count = cresp.accepted_count;
        return true;
    }

    split_gen3_mid_resp mid{};
    split_gen_pipe_trace("entry", phase, "recv_mid_enter", n_tokens, n_embd, pos_start, -1, layer_end);
    if (!split_gen3_recv_mid_resp(peer_fd, mid)) {
        fprintf(stderr, "gen3_a: recv mid resp failed\n");
        return false;
    }
    split_gen_pipe_trace("entry", phase, "recv_mid_exit", n_tokens, n_embd, pos_start, -1, layer_end);

    resp.token_id       = mid.token_id;
    resp.ms_b_compute   = mid.ms_b_compute;
    resp.ms_bc_xfer     = mid.ms_bc_xfer;
    resp.ms_c_compute   = mid.ms_c_compute;
    resp.ms_c_sample    = mid.ms_c_sample;
    resp.accepted_count = mid.accepted_count;
    return true;
}

static bool send_hidden_to_b_only(
        int peer_fd,
        llama_context * ctx,
        int32_t n_tokens,
        int32_t n_embd,
        int32_t layer_end,
        int32_t pos_start,
        bool include_logits,
        int32_t debug_step,
        const char * phase,
        trace_recorder * dbg,
        double ms_a,
        double & ms_ab_xfer_out) {
    perf_trace_refresh_context();
    perf_trace_set_wave_id(perf_trace_wave_id_from_step(phase, debug_step));
    const int32_t tok_idx = (phase && std::strcmp(phase, "decode") == 0) ? debug_step : -1;

    std::vector<float> hidden_buf;
    hidden_pack_stats pack_stats{};
    if (perf_trace_enabled()) {
        if (phase && std::strcmp(phase, "decode") == 0) {
            perf_trace_ensure_decode_context(tok_idx, perf_trace_wave_id_from_step(phase, debug_step));
        } else {
            perf_trace_set_wave_id(perf_trace_wave_id_from_step(phase, debug_step));
        }
    }
    if (!hidden_pack_gather_stage_hidden(ctx, n_tokens, n_embd, hidden_buf, pack_stats.gather)) {
        fprintf(stderr, "gen3_a: missing hidden state\n");
        return false;
    }
    const float * hidden = hidden_buf.data();
    const int32_t payload_bytes = n_tokens * n_embd * (int32_t) sizeof(float);

    hidden_transport_trace tr_send = dist_debug_transport_send(
            debug_step, phase, "ab", n_tokens, n_embd, layer_end, pos_start, hidden, 0.0);
    if (dbg) {
        dbg->emit_transport(tr_send);
    }

    split_gen_pipe_trace("entry", phase, "send_hidden_enter", n_tokens, n_embd, pos_start, -1, layer_end);
    if (!hidden_pack_send_ab_hidden(peer_fd, n_tokens, n_embd, layer_end, pos_start,
            include_logits ? 1 : 0, hidden, pack_stats.send)) {
        fprintf(stderr, "gen3_a: send hidden failed\n");
        return false;
    }
    ms_ab_xfer_out = pack_stats.send.send_us / 1000.0;
    split_gen_pipe_trace("entry", phase, "send_hidden_exit", n_tokens, n_embd, pos_start, -1, layer_end);

    if (perf_trace_enabled()) {
        pack_stats.pack_total_us = pack_stats.gather.alloc_us + pack_stats.gather.sync_us
                + pack_stats.gather.gather_us
                + pack_stats.gather.copy_us + pack_stats.gather.serialize_us
                + pack_stats.send.frame_us + pack_stats.send.send_us;
        hidden_pack_emit_breakdown_spans(pack_stats, tok_idx, payload_bytes);
    }
    (void) ms_a;
    return true;
}

struct entry_ab_pending {
    split_gen_a_req req{};
    int32_t         wave_id       = -1;
    int32_t         debug_step    = 0;
    bool            decode_step   = false;
    bool            include_logits = false;
    double          ms_a          = 0.0;
    double          ms_ab_xfer    = 0.0;
};

// Sizes the entry-side bounded wait for a draft off recent arrival latency
// instead of a fixed constant, since a Wi-Fi link's RTT/jitter can swing
// from ~5ms to 100ms+ over seconds. Tracks the last HISTORY (pos_start,
// latency) samples and recomputes wait_window_ms as clamp(percentile +
// margin, min, max) every RECOMPUTE_EVERY samples (not on every one, to
// avoid the window itself oscillating). wait_window_ms is the only thing
// callers read; the source of that value can later be swapped for an
// orchestrator-supplied estimate without touching the caller.
//
// Which percentile (or a fixed window) is used is set once at startup from
// SPEC_WAIT_POLICY, so the policy can be A/B'd against a live cluster
// without a rebuild between runs -- see split_gen3_a's entry point for the
// env var parsing. Default (unset) is p80: the 2026-07-21 baseline bench
// (docs/TASK_19_SPECULATIVE_PIPELINE_STUDY.md section H) found p80 beat
// p95 on throughput in both rounds, including once under a *worse*
// measured network -- the one comparison in that dataset that survived
// the network confound. fixed:8 (the original hardcoded window) was
// confirmed suboptimal.
struct draft_wait_estimator {
    static constexpr size_t HISTORY         = 128;
    static constexpr size_t RECOMPUTE_EVERY = 32;
    static constexpr double MIN_WAIT_MS     = 4.0;
    static constexpr double MAX_WAIT_MS     = 30.0;
    static constexpr double SAFETY_MARGIN_MS = 2.0;

    bool                  fixed_policy = false;
    double                percentile   = 0.80; // used when !fixed_policy

    std::mutex           mu;
    std::vector<double>  samples;
    size_t               next_slot       = 0;
    size_t               since_recompute = 0;
    size_t               hits            = 0;
    size_t               misses          = 0;
    std::atomic<double>  wait_window_ms{8.0}; // seed = old fixed value

    // ms is the observed wait needed for a hit, or the window used for a
    // miss (a censored lower bound -- we don't know how much longer the
    // draft would have taken, only that it was more than we waited).
    void add_sample_ms(double ms, bool hit) {
        std::lock_guard<std::mutex> lock(mu);
        if (hit) { ++hits; } else { ++misses; }
        if (samples.size() < HISTORY) {
            samples.push_back(ms);
        } else {
            samples[next_slot] = ms;
            next_slot = (next_slot + 1) % HISTORY;
        }
        if (++since_recompute < RECOMPUTE_EVERY || samples.size() < 8) {
            return;
        }
        since_recompute = 0;
        std::vector<double> sorted = samples;
        std::sort(sorted.begin(), sorted.end());
        const double p50 = sorted[(size_t) (0.50 * (sorted.size() - 1))];
        const double p95 = sorted[(size_t) (0.95 * (sorted.size() - 1))];
        // Always measured off p95 regardless of policy, so every policy's
        // log is directly comparable in the A/B bench even when the
        // window itself isn't derived from p95.
        if (!fixed_policy) {
            const double target = sorted[(size_t) (percentile * (sorted.size() - 1))];
            const double window = std::clamp(target + SAFETY_MARGIN_MS, MIN_WAIT_MS, MAX_WAIT_MS);
            wait_window_ms.store(window);
        }
        const double hit_rate = (hits + misses) > 0
                ? 100.0 * (double) hits / (double) (hits + misses) : 0.0;
        fprintf(stderr, "SPEC_DEBUG entry: wait_window=%.1fms arrival_p50=%.1fms arrival_p95=%.1fms hit_rate=%.0f%%\n",
                wait_window_ms.load(), p50, p95, hit_rate);
        hits = 0;
        misses = 0;
    }
};

// Parses SPEC_WAIT_POLICY: "fixed:<ms>", "p50"/"p80"/"p90"/"p95"/"p99", or
// unset (defaults to p80, the current best-known policy).
static void configure_wait_policy_from_env(draft_wait_estimator & est) {
    const char * raw = getenv("SPEC_WAIT_POLICY");
    if (!raw || !*raw) {
        return; // keep defaults: !fixed_policy, percentile=0.95
    }
    const std::string policy(raw);
    if (policy.rfind("fixed:", 0) == 0) {
        const double ms = std::atof(policy.c_str() + 6);
        est.fixed_policy = true;
        est.wait_window_ms.store(ms);
        fprintf(stderr, "gen3_a: SPEC_WAIT_POLICY=fixed:%.1fms\n", ms);
        return;
    }
    static const std::vector<std::pair<std::string, double>> named = {
        {"p50", 0.50}, {"p80", 0.80}, {"p90", 0.90}, {"p95", 0.95}, {"p99", 0.99},
    };
    for (const auto & kv : named) {
        if (policy == kv.first) {
            est.fixed_policy = false;
            est.percentile = kv.second;
            fprintf(stderr, "gen3_a: SPEC_WAIT_POLICY=%s\n", kv.first.c_str());
            return;
        }
    }
    fprintf(stderr, "gen3_a: unrecognized SPEC_WAIT_POLICY=%s, falling back to p80\n", raw);
}

// Draft token ids delivered directly from the node holding `final` over the
// fa-link (Task 19 Phase 3, placement per TASK_19_SPECULATIVE_PIPELINE_STUDY
// §A). Single-slot: final only ever has one outstanding draft (for the
// position right after the token it just produced).
struct fa_draft_buffer {
    std::mutex              mu;
    std::condition_variable cv;
    int32_t              pos = -1;
    std::vector<int32_t> ids;
    // Timestamp (ggml_time_us) of the last pos/ids write, set under mu by
    // the fa-link receiver thread. Read by the consumer to measure how
    // long it actually waited for a given draft (see draft_wait_estimator).
    int64_t              arrival_us = 0;
    // Set once final has connected in; used to ship the prompt tokens at
    // PREFILL time so the draft model can prime its own KV cache (see
    // split_gen3_c.cpp's fa_prime receiver thread). Same fd the receiver
    // thread above reads from -- TCP is full duplex, and each direction
    // only ever carries one message shape, so this is unambiguous.
    int                  fd  = -1;
    draft_wait_estimator wait_est;
};

struct entry_stage_state {
    llama_context * ctx          = nullptr;
    int             b_fd         = -1;
    int             ctrl_fd      = -1;
    int             layer_start  = 0;
    int             layer_end    = 0;
    int             n_embd       = 0;
    int             n_vocab      = 0;
    bool            next_is_final = false;
    trace_recorder * dbg         = nullptr;
    int32_t *       debug_step   = nullptr;
    int32_t *       entry_queue_depth = nullptr;
    // Non-null only in the queued-session path, where a separate receiver
    // thread can deliver RESET concurrently with the consumer thread's
    // decode on the same ctx. Guards both against that race.
    std::mutex *    ctx_mu       = nullptr;
    int32_t         next_pos     = 0;
    // Non-null only when --fa-port was given (speculative mode enabled).
    fa_draft_buffer * fa_buf     = nullptr;
};

static bool entry_process_work_item(
        entry_stage_state & st,
        wave_work_item & item,
        const bool send_early_token,
        std::mutex * ctrl_mu,
        const bool pipeline_ab,
        std::mutex * ab_mu,
        std::deque<entry_ab_pending> * ab_pending) {
    split_gen_a_req & req = item.req;
    std::vector<int32_t> & tokens_i32 = item.tokens;
    std::vector<float> & hidden_in = item.hidden;
    const int32_t hidden_n_embd = item.hidden_n_embd;
    const int ctrl_fd = st.ctrl_fd;

    // Speculative mode: the client sends only the anchor token; extend the
    // wave here with whatever final has already drafted for this position
    // (delivered ahead of time over the fa-link). No buffered draft (not
    // ready yet, or speculation disabled) just leaves a 1-token wave, which
    // final_verify_wave treats as plain decode.
    if (req.cmd == SPLIT_GEN_CMD_VERIFY && st.fa_buf != nullptr && tokens_i32.size() == 1) {
        std::unique_lock<std::mutex> lock(st.fa_buf->mu);
        // The draft ships from final only after its k decode steps, racing
        // the token's own longer return trip (final->middle->entry->client
        // ->entry); it loses that race by a hair more often than not. A
        // short bounded wait flips most of those photo finishes: one
        // accepted token repays a few ms of waiting many times over. The
        // window itself is adaptive (see draft_wait_estimator) since a
        // Wi-Fi link's RTT can swing an order of magnitude within seconds.
        const int64_t t_need_us = ggml_time_us();
        const double  window_ms = st.fa_buf->wait_est.wait_window_ms.load();
        if (st.fa_buf->pos != req.pos_start || st.fa_buf->ids.empty()) {
            st.fa_buf->cv.wait_for(lock, std::chrono::duration<double, std::milli>(window_ms), [&] {
                return st.fa_buf->pos == req.pos_start && !st.fa_buf->ids.empty();
            });
        }
        const bool hit = st.fa_buf->pos == req.pos_start && !st.fa_buf->ids.empty();
        const double latency_ms = hit
                ? std::max(0.0, (double) (st.fa_buf->arrival_us - t_need_us) / 1000.0)
                : window_ms; // censored: draft took at least this long (or never came)
        st.fa_buf->wait_est.add_sample_ms(latency_ms, hit);
        fprintf(stderr, "SPEC_DEBUG entry: pos=%d draft_%s window=%.1fms\n", req.pos_start, hit ? "hit" : "miss", window_ms);
        if (hit) {
            tokens_i32.insert(tokens_i32.end(), st.fa_buf->ids.begin(), st.fa_buf->ids.end());
            req.n_tokens = (int32_t) tokens_i32.size();
        }
        st.fa_buf->pos = -1;
        st.fa_buf->ids.clear();
    }
    const int32_t drafted_n = req.cmd == SPLIT_GEN_CMD_VERIFY && !tokens_i32.empty()
            ? (int32_t) tokens_i32.size() - 1 : 0;

    // Prime final's draft model with the real prompt (best-effort: if final
    // hasn't connected its fa-link yet, this session just runs without
    // speculation -- see F.1/F.2 in TASK_19_SPECULATIVE_PIPELINE_STUDY.md).
    if (req.cmd == SPLIT_GEN_CMD_PREFILL && st.fa_buf != nullptr) {
        int prime_fd = -1;
        {
            std::lock_guard<std::mutex> lock(st.fa_buf->mu);
            prime_fd = st.fa_buf->fd;
        }
        if (prime_fd >= 0) {
            split_ab_send_verify_ids(prime_fd, 0, tokens_i32.data(), (int32_t) tokens_i32.size());
        }
    }

    const int64_t t0 = ggml_time_us();
    int decode_rc = 0;
    const char * phase = (req.cmd == SPLIT_GEN_CMD_PREFILL ||
                          req.cmd == SPLIT_GEN_CMD_PREFILL_HIDDEN ||
                          req.cmd == SPLIT_GEN_CMD_VERIFY) ? "prefill" : "decode";
    const int32_t in_tok = tokens_i32.empty() ? -1 : tokens_i32[0];
    const int32_t wave_id = item.wave_id >= 0
            ? item.wave_id
            : perf_trace_wave_id_from_step(phase, st.debug_step ? *st.debug_step : 0);

    if (st.dbg) {
        st.dbg->emit_step_begin(st.debug_step ? *st.debug_step : 0, phase, in_tok, req.pos_start, 0);
        dist_debug_log_position(st.dbg, st.debug_step ? *st.debug_step : 0, phase, st.ctx, req.pos_start,
                req.cmd == SPLIT_GEN_CMD_PREFILL || req.cmd == SPLIT_GEN_CMD_PREFILL_HIDDEN
                        ? req.n_tokens : 1,
                1);
    }

    split_gen_pipe_trace("entry", phase, "decode_enter", req.n_tokens, hidden_n_embd > 0 ? hidden_n_embd : st.n_embd,
            req.pos_start, st.layer_start, st.layer_end);
    const int32_t tok_idx = (phase && std::strcmp(phase, "decode") == 0 && st.debug_step)
            ? *st.debug_step : -1;
    const bool decode_step = (phase && std::strcmp(phase, "decode") == 0);
    if (decode_step && perf_trace_enabled()) {
        perf_trace_ensure_decode_context(tok_idx, wave_id);
        perf_trace_set_component("entry");
        perf_emit_instant("ENTRY_RECEIVE", perf_category::NETWORK, "entry", tok_idx, nullptr);
        if (st.entry_queue_depth) {
            perf_emit_queue_depth("entry", tok_idx, *st.entry_queue_depth);
            *st.entry_queue_depth = std::max(0, *st.entry_queue_depth - 1);
        }
    }
    if (perf_trace_enabled() && decode_step) {
        perf_trace_ensure_decode_context(tok_idx, wave_id);
    }
    perf_span compute_span("ENTRY_COMPUTE_BEGIN", "ENTRY_COMPUTE_END", perf_category::COMPUTE, "entry");
    compute_span.set_token_idx(tok_idx);
    split_gen_rollback_kv(st.ctx, st.next_pos, req.pos_start, "gen3_a");
    if (req.cmd == SPLIT_GEN_CMD_PREFILL || req.cmd == SPLIT_GEN_CMD_VERIFY) {
        if (st.layer_start > 0) {
            fprintf(stderr, "gen3_a: token prefill requires layer_start=0\n");
            return false;
        }
        std::vector<llama_token> tokens((size_t) req.n_tokens);
        for (int32_t i = 0; i < req.n_tokens; ++i) {
            tokens[i] = (llama_token) tokens_i32[i];
        }
        decode_rc = split_gen_decode_tokens(st.ctx, tokens, req.pos_start, true);
    } else if (req.cmd == SPLIT_GEN_CMD_DECODE) {
        if (st.layer_start > 0) {
            fprintf(stderr, "gen3_a: token decode requires layer_start=0\n");
            return false;
        }
        decode_rc = split_gen_decode_one(st.ctx, (llama_token) tokens_i32[0], req.pos_start);
    } else if (req.cmd == SPLIT_GEN_CMD_PREFILL_HIDDEN) {
        const int32_t n_emb = hidden_n_embd > 0 ? hidden_n_embd : st.n_embd;
        decode_rc = split_gen_decode_hidden(
                st.ctx, hidden_in.data(), req.n_tokens, n_emb, req.pos_start, true, st.layer_start > 0);
    } else if (req.cmd == SPLIT_GEN_CMD_DECODE_HIDDEN) {
        const int32_t n_emb = hidden_n_embd > 0 ? hidden_n_embd : st.n_embd;
        decode_rc = split_gen_decode_hidden(
                st.ctx, hidden_in.data(), 1, n_emb, req.pos_start, true, st.layer_start > 0);
    } else {
        fprintf(stderr, "gen3_a: unknown cmd %u\n", req.cmd);
        return false;
    }
    split_gen_pipe_trace("entry", phase, "decode_exit", req.n_tokens, hidden_n_embd > 0 ? hidden_n_embd : st.n_embd,
            req.pos_start, st.layer_start, st.layer_end);

    const double ms_a = (ggml_time_us() - t0) / 1000.0;
    if (decode_rc != 0) {
        fprintf(stderr, "gen3_a: decode failed cmd=%u\n", req.cmd);
        return false;
    }

    const int32_t n_out = (req.cmd == SPLIT_GEN_CMD_PREFILL ||
                           req.cmd == SPLIT_GEN_CMD_PREFILL_HIDDEN ||
                           req.cmd == SPLIT_GEN_CMD_VERIFY) ? req.n_tokens : 1;
    const int32_t le = req.layer_end > 0 ? req.layer_end : st.layer_end;
    st.next_pos = req.pos_start + n_out;

    if (req.cmd == SPLIT_GEN_CMD_VERIFY) {
        if (pipeline_ab) {
            fprintf(stderr, "gen3_a: verify wave requires DIST_RUNTIME_ENTRY_QUEUE=0\n");
            return false;
        }
        if (!split_ab_send_verify_ids(st.b_fd, req.pos_start,
                    tokens_i32.data(), (int32_t) tokens_i32.size())) {
            fprintf(stderr, "gen3_a: send verify ids failed\n");
            return false;
        }
    }

    if (st.dbg) {
        dist_debug_log_kv(st.dbg, st.debug_step ? *st.debug_step : 0, phase, st.ctx, 0);
        dist_debug_log_hidden_out(st.dbg, st.debug_step ? *st.debug_step : 0, phase, st.ctx, n_out, st.n_embd, "entry");
        std::vector<int32_t> pos_buf((size_t) n_out);
        std::vector<int32_t> tok_buf((size_t) n_out);
        for (int32_t i = 0; i < n_out; ++i) {
            pos_buf[(size_t) i] = req.pos_start + i;
            tok_buf[(size_t) i] = (i < (int32_t) tokens_i32.size()) ? tokens_i32[(size_t) i] : -1;
        }
        dist_debug_log_runtime_state(
                st.dbg, st.debug_step ? *st.debug_step : 0, phase, "entry", st.ctx, n_out,
                tok_buf.data(), pos_buf.data(), -1, in_tok);
    }

    split_gen3_a_resp resp{};
    std::vector<float> logits;

    if (pipeline_ab && ab_mu != nullptr && ab_pending != nullptr) {
        entry_ab_pending pending;
        pending.req            = req;
        pending.wave_id        = wave_id;
        pending.debug_step     = st.debug_step ? *st.debug_step : 0;
        pending.decode_step    = decode_step;
        pending.include_logits = req.include_logits != 0;
        pending.ms_a           = ms_a;
        pending.ms_ab_xfer     = 0.0;
        {
            std::lock_guard<std::mutex> lock(*ab_mu);
            ab_pending->push_back(pending);
        }
        double ms_ab_xfer = 0.0;
        if (!send_hidden_to_b_only(st.b_fd, st.ctx, n_out, st.n_embd, le, req.pos_start,
                    req.include_logits != 0, st.debug_step ? *st.debug_step : 0, phase, st.dbg,
                    ms_a, ms_ab_xfer)) {
            std::lock_guard<std::mutex> lock(*ab_mu);
            if (!ab_pending->empty()) {
                ab_pending->pop_back();
            }
            return false;
        }
        {
            std::lock_guard<std::mutex> lock(*ab_mu);
            if (!ab_pending->empty()) {
                ab_pending->back().ms_ab_xfer = ms_ab_xfer;
            }
        }
    } else {
        if (!forward_to_peer(st.b_fd, st.ctx, n_out, st.n_embd, le, req.pos_start,
                req.include_logits != 0, st.n_vocab, ms_a, st.next_is_final,
                st.debug_step ? *st.debug_step : 0, phase, st.dbg, resp, logits)) {
            return false;
        }

        if (req.cmd == SPLIT_GEN_CMD_VERIFY && resp.accepted_count > 0) {
            const int32_t n_ids = std::min({ resp.accepted_count, drafted_n, SPLIT_GEN_SPEC_MAX_K });
            for (int32_t i = 0; i < n_ids; ++i) {
                resp.accepted_ids[i] = tokens_i32[(size_t) i + 1];
            }
        }

        if (send_early_token && ctrl_mu != nullptr) {
            std::lock_guard<std::mutex> lock(*ctrl_mu);
            if (!split_gen_send_token_ready(ctrl_fd, resp.token_id, req.pos_start, wave_id)) {
                return false;
            }
        }

        if (ctrl_mu != nullptr) {
            std::lock_guard<std::mutex> lock(*ctrl_mu);
            if (!split_gen3_send_a_resp(ctrl_fd, resp, logits.empty() ? nullptr : logits.data(), st.n_vocab)) {
                fprintf(stderr, "gen3_a: send resp failed\n");
                return false;
            }
        } else if (!split_gen3_send_a_resp(ctrl_fd, resp, logits.empty() ? nullptr : logits.data(), st.n_vocab)) {
            fprintf(stderr, "gen3_a: send resp failed\n");
            return false;
        }

        if (st.dbg && resp.token_id >= 0) {
            st.dbg->emit_token_selected(st.debug_step ? *st.debug_step : 0, phase, resp.token_id, req.pos_start, false);
        }
    }

    if (decode_step && st.entry_queue_depth) {
        (*st.entry_queue_depth)++;
    }
    if (st.debug_step) {
        (*st.debug_step)++;
    }
    return true;
}

static bool entry_handle_control_cmd(
        const entry_stage_state & st,
        const split_gen_a_req & req,
        int32_t & debug_step,
        bool & normal_shutdown) {
    const int ctrl_fd = st.ctrl_fd;
    const int b_fd = st.b_fd;

    if (req.cmd == SPLIT_GEN_CMD_PROTO_NEGOTIATE) {
        return runtime_protocol_handle_negotiate_server(ctrl_fd, req);
    }
    if (req.cmd == SPLIT_GEN_CMD_SHUTDOWN) {
        split_ab_send_shutdown(b_fd);
        split_gen3_a_resp resp{};
        resp.magic   = SPLIT_GEN_MAGIC;
        resp.version = SPLIT_GEN3_VERSION;
        split_gen3_send_a_resp(ctrl_fd, resp, nullptr, 0);
        normal_shutdown = true;
        return true;
    }
    if (req.cmd == SPLIT_GEN_CMD_RESET) {
        if (st.ctx_mu) {
            st.ctx_mu->lock();
        }
        llama_memory_clear(llama_get_memory(st.ctx), true);
        llama_clear_hidden_state(st.ctx);
        if (st.ctx_mu) {
            st.ctx_mu->unlock();
        }
        debug_step = 0;
        if (st.dbg) {
            st.dbg->emit_step_begin(0, "reset", -1, 0, 0);
        }
        split_ab_send_reset(b_fd);
        split_gen3_a_resp resp{};
        resp.magic    = SPLIT_GEN_MAGIC;
        resp.version  = SPLIT_GEN3_VERSION;
        resp.token_id = -1;
        split_gen3_send_a_resp(ctrl_fd, resp, nullptr, 0);
        return true;
    }
    return false;
}

static bool entry_run_queued_session(
        entry_stage_state st,
        wave_inbound_queue & queue,
        int32_t & debug_step,
        bool & normal_shutdown,
        bool & pipe_failed) {
    std::mutex ctrl_mu;
    std::mutex ab_mu;
    std::deque<entry_ab_pending> ab_pending;
    std::deque<split_gen3_a_resp> deferred_completes;
    std::mutex deferred_mu;
    std::atomic<bool> stop{ false };
    std::atomic<int> processor_active{ 0 };
    int32_t entry_queue_depth = 0;
    st.debug_step = &debug_step;
    st.entry_queue_depth = &entry_queue_depth;
    // See entry_stage_state::ctx_mu: the receiver thread's RESET handling and
    // this function's consumer loop both touch st.ctx concurrently.
    std::mutex ctx_mu;
    st.ctx_mu = &ctx_mu;
    const bool pipeline_ab = runtime_stage_queue_enabled();
    const bool defer_complete = pipeline_ab && runtime_client_pipeline_enabled();

    auto flush_deferred_complete = [&]() {
        split_gen3_a_resp deferred{};
        {
            std::lock_guard<std::mutex> dlock(deferred_mu);
            if (deferred_completes.empty()) {
                return;
            }
            deferred = deferred_completes.front();
            deferred_completes.pop_front();
        }
        std::lock_guard<std::mutex> lock(ctrl_mu);
        if (!split_gen3_send_a_resp(st.ctrl_fd, deferred, nullptr, st.n_vocab)) {
            fprintf(stderr, "gen3_a: deferred complete failed\n");
            pipe_failed = true;
            stop.store(true);
        }
    };

    std::thread ab_responder;
    if (pipeline_ab) {
        ab_responder = std::thread([&] {
            while (!stop.load() || !ab_pending.empty()) {
                entry_ab_pending pending;
                {
                    std::unique_lock<std::mutex> lock(ab_mu);
                    while (!stop.load() && ab_pending.empty()) {
                        lock.unlock();
                        std::this_thread::sleep_for(std::chrono::milliseconds(1));
                        lock.lock();
                    }
                    if (ab_pending.empty()) {
                        break;
                    }
                    pending = ab_pending.front();
                    ab_pending.pop_front();
                }

                split_gen3_a_resp resp{};
                resp.magic          = SPLIT_GEN_MAGIC;
                resp.version        = SPLIT_GEN3_VERSION;
                resp.include_logits = 0;
                resp.n_vocab        = st.n_vocab;

                if (st.next_is_final) {
                    split_gen3_c_resp cresp{};
                    if (!split_gen3_recv_c_resp(st.b_fd, cresp)) {
                        if (!stop.load()) {
                            pipe_failed = true;
                        }
                        stop.store(true);
                        break;
                    }
                    resp.token_id     = cresp.token_id;
                    resp.ms_a_compute = pending.ms_a;
                    resp.ms_ab_xfer   = pending.ms_ab_xfer;
                    resp.ms_b_compute = 0.0;
                    resp.ms_bc_xfer   = 0.0;
                    resp.ms_c_compute = cresp.ms_compute;
                    resp.ms_c_sample  = cresp.ms_sample;
                } else {
                    split_gen3_mid_resp mid{};
                    if (!split_gen3_recv_mid_resp(st.b_fd, mid)) {
                        if (!stop.load()) {
                            pipe_failed = true;
                        }
                        stop.store(true);
                        break;
                    }
                    resp.token_id     = mid.token_id;
                    resp.ms_a_compute = pending.ms_a;
                    resp.ms_ab_xfer   = pending.ms_ab_xfer;
                    resp.ms_b_compute = mid.ms_b_compute;
                    resp.ms_bc_xfer   = mid.ms_bc_xfer;
                    resp.ms_c_compute = mid.ms_c_compute;
                    resp.ms_c_sample  = mid.ms_c_sample;
                }

                const char * phase = pending.decode_step ? "decode" : "prefill";
                if (st.dbg) {
                    dist_debug_log_runtime_state(
                            st.dbg, pending.debug_step, phase, "entry", st.ctx, 1,
                            nullptr, nullptr, -1, resp.token_id);
                    if (resp.token_id >= 0) {
                        st.dbg->emit_token_selected(
                                pending.debug_step, phase, resp.token_id, pending.req.pos_start, false);
                    }
                }

                {
                    std::lock_guard<std::mutex> lock(ctrl_mu);
                    if (!split_gen_send_token_ready(
                                st.ctrl_fd, resp.token_id, pending.req.pos_start, pending.wave_id)) {
                        pipe_failed = true;
                        stop.store(true);
                        break;
                    }
                    std::lock_guard<std::mutex> dlock(deferred_mu);
                    deferred_completes.push_back(resp);
                }
                if (!defer_complete || !pending.decode_step) {
                    flush_deferred_complete();
                }
            }
        });
    }

    std::thread receiver([&] {
        while (!stop.load()) {
            split_gen_a_req req{};
            std::vector<int32_t> tokens_i32;
            std::vector<float> hidden_in;
            int32_t hidden_n_embd = 0;
            perf_trace_sched_queue_wait_begin();
            const bool got_req = split_gen_recv_req(st.ctrl_fd, req, tokens_i32, &hidden_in, &hidden_n_embd);
            perf_trace_sched_queue_wait_end();
            if (!got_req) {
                stop.store(true);
                break;
            }

            if (req.cmd == SPLIT_GEN_CMD_PROTO_NEGOTIATE ||
                    req.cmd == SPLIT_GEN_CMD_SHUTDOWN ||
                    req.cmd == SPLIT_GEN_CMD_RESET) {
                int32_t local_step = debug_step;
                bool shutdown = false;
                if (!entry_handle_control_cmd(st, req, local_step, shutdown)) {
                    stop.store(true);
                    break;
                }
                debug_step = local_step;
                if (shutdown) {
                    normal_shutdown = true;
                    stop.store(true);
                    break;
                }
                continue;
            }

            if (req.cmd == SPLIT_GEN_CMD_DRAIN_PENDING) {
                flush_deferred_complete();
                continue;
            }

            const char * phase = (req.cmd == SPLIT_GEN_CMD_PREFILL ||
                                  req.cmd == SPLIT_GEN_CMD_PREFILL_HIDDEN) ? "prefill" : "decode";
            wave_work_item item;
            item.req           = req;
            item.tokens        = std::move(tokens_i32);
            item.hidden        = std::move(hidden_in);
            item.hidden_n_embd = hidden_n_embd;
            item.wave_id       = perf_trace_wave_id_from_step(phase, debug_step);
            const int32_t queued_wave_id = item.wave_id;

            const int32_t ack_depth = queue.observable_depth(true) + 1;
            {
                std::lock_guard<std::mutex> lock(ctrl_mu);
                if (!split_gen_send_queue_ack(st.ctrl_fd, ack_depth, queued_wave_id, 0)) {
                    stop.store(true);
                    break;
                }
            }
            if (defer_complete) {
                flush_deferred_complete();
            }

            queue.push(std::move(item));
            entry_queue_depth = ack_depth;
            if (perf_trace_enabled()) {
                perf_trace_set_wave_id(queued_wave_id);
                perf_emit_instant("WAVE_QUEUED", perf_category::WAIT, "entry", -1, nullptr);
                perf_emit_queue_depth("entry", -1, entry_queue_depth);
            }
        }
        // The consumer below may be parked in queue.pop() with nothing left
        // to wake it: after this thread exits, push() never fires again, so
        // an un-closed queue leaves the consumer -- and therefore this whole
        // session function -- blocked forever. The ctrl accept loop in
        // main() then never runs again, every later connect sees no reply
        // to its negotiate request, and the orchestrator burns ~30s per
        // generate on a full pipeline recovery. Always close on exit.
        queue.close();
    });

    while (!stop.load() || queue.depth() > 0) {
        wave_work_item item;
        if (!queue.try_pop(item)) {
            if (stop.load()) {
                break;
            }
            if (!queue.pop(item)) {
                // Queue closed by the receiver and fully drained.
                break;
            }
        }
        processor_active.fetch_add(1);
        ctx_mu.lock();
        const bool item_ok = entry_process_work_item(
                    st, item, true, &ctrl_mu, pipeline_ab,
                    pipeline_ab ? &ab_mu : nullptr,
                    pipeline_ab ? &ab_pending : nullptr);
        ctx_mu.unlock();
        if (!item_ok) {
            pipe_failed = true;
            stop.store(true);
            processor_active.fetch_sub(1);
            break;
        }
        processor_active.fetch_sub(1);
        if (pipeline_ab && queue.depth() == 0 && processor_active.load() == 0) {
            flush_deferred_complete();
        }
    }

    while (pipeline_ab) {
        bool has_deferred = false;
        {
            std::lock_guard<std::mutex> dlock(deferred_mu);
            has_deferred = !deferred_completes.empty();
        }
        if (!has_deferred) {
            break;
        }
        flush_deferred_complete();
    }

    stop.store(true);
    // Mirror of the receiver-side close: if the receiver is parked in a
    // blocking push() against a full queue when this consumer dies, only a
    // close() wakes it so the join below can complete.
    queue.close();
    if (receiver.joinable()) {
        receiver.join();
    }
    if (ab_responder.joinable()) {
        ab_responder.join();
    }
    return !pipe_failed;
}

int main(int argc, char ** argv) {
    split_tcp_init();
    if (argc < 2) {
        usage(argv[0]);
        return 1;
    }

    const char * model_path = argv[1];
    int ctrl_port = -1;
    int b_port    = -1;
    int fa_port   = -1;
    int layer_start = 0;
    int layer_end = 5;
    bool next_is_final = false;
    const char * b_host = "127.0.0.1";
    const char * bind_host = "0.0.0.0";
    std::string ready_file;

    for (int i = 2; i < argc; ++i) {
        if (strcmp(argv[i], "--ctrl-port") == 0 && i + 1 < argc) {
            ctrl_port = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--b-port") == 0 && i + 1 < argc) {
            b_port = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--b-host") == 0 && i + 1 < argc) {
            b_host = argv[++i];
        } else if (strcmp(argv[i], "--fa-port") == 0 && i + 1 < argc) {
            fa_port = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--bind") == 0 && i + 1 < argc) {
            bind_host = argv[++i];
        } else if (strcmp(argv[i], "--layer-start") == 0 && i + 1 < argc) {
            layer_start = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--layer-end") == 0 && i + 1 < argc) {
            layer_end = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--next-final") == 0) {
            next_is_final = true;
        } else if (strcmp(argv[i], "--ready-file") == 0 && i + 1 < argc) {
            ready_file = argv[++i];
        } else {
            usage(argv[0]);
            return 1;
        }
    }

    if (ctrl_port < 0 || b_port < 0) {
        usage(argv[0]);
        return 1;
    }

    split_gen_write_ready_state(ready_file, "MODEL_LOADING");

    const int listen_fd = split_tcp_listen_host(bind_host, ctrl_port);
    if (listen_fd < 0) {
        split_gen_write_ready_state(ready_file, "FAILED");
        fprintf(stderr, "gen3_a: ctrl listen failed port=%d\n", ctrl_port);
        return 1;
    }

    // Task 19 Phase 3: direct link from whichever node holds `final`, used
    // to deliver drafted token ids ahead of the client's next request (see
    // TASK_19_SPECULATIVE_PIPELINE_STUDY.md SA). Optional: only set up when
    // the orchestrator allocated a port for this session.
    fa_draft_buffer fa_buf;
    configure_wait_policy_from_env(fa_buf.wait_est);
    std::thread     fa_thread;
    if (fa_port > 0) {
        const int fa_listen_fd = split_tcp_listen_host(bind_host, fa_port);
        if (fa_listen_fd < 0) {
            fprintf(stderr, "gen3_a: fa listen failed port=%d (speculative mode disabled)\n", fa_port);
        } else {
            fa_thread = std::thread([fa_listen_fd, &fa_buf]() {
                const int fa_fd = split_tcp_accept(fa_listen_fd);
                if (fa_fd < 0) {
                    fprintf(stderr, "gen3_a: fa accept failed\n");
                    return;
                }
                {
                    std::lock_guard<std::mutex> lock(fa_buf.mu);
                    fa_buf.fd = fa_fd;
                }
                while (true) {
                    split_ab_cmd cmd;
                    if (!split_ab_recv_cmd(fa_fd, cmd) || cmd != SPLIT_AB_CMD_VERIFY_IDS) {
                        break;
                    }
                    int32_t pos = -1;
                    std::vector<int32_t> ids;
                    if (!split_ab_recv_verify_ids(fa_fd, pos, ids)) {
                        break;
                    }
                    {
                        std::lock_guard<std::mutex> lock(fa_buf.mu);
                        fa_buf.pos = pos;
                        fa_buf.ids = std::move(ids);
                        fa_buf.arrival_us = ggml_time_us();
                    }
                    fa_buf.cv.notify_all();
                }
            });
            fa_thread.detach();
        }
    }

    split_gen_write_ready_state(ready_file, "LISTENER_READY");

    ggml_backend_load_all();

    std::string load_err;
    llama_model * model = split_gen_load_model(model_path, load_err);
    if (!model) {
        split_tcp_close(listen_fd);
        split_gen_write_ready_state(ready_file, "FAILED");
        fprintf(stderr, "gen3_a: load model failed: %s\n", load_err.c_str());
        return 1;
    }

    const int32_t n_embd  = llama_model_n_embd(model);
    const int32_t n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model));

    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx   = 512;
    cparams.n_batch = 512;
    // n_ubatch > 1 so a k+1-token verify wave decodes as ONE graph instead
    // of k+1 sequential single-token graphs -- same fix as final/middle
    // (split_gen3_c.cpp, split_gen3_b.cpp). Hidden-state gather already
    // reads by index (llama_get_embeddings_ith), which is ubatch-agnostic,
    // so no other change is needed here.
    cparams.n_ubatch = 32;
    cparams.no_perf = true;
    cparams.layer_start = layer_start;
    cparams.layer_end   = layer_end;
    fprintf(stderr, "gen3_a: create context layer_start=%d layer_end=%d\n", layer_start, layer_end);
    llama_context * ctx = llama_init_from_model(model, cparams);
    if (!ctx) {
        split_tcp_close(listen_fd);
        split_gen_write_ready_state(ready_file, "FAILED");
        fprintf(stderr, "gen3_a: create context failed\n");
        return 1;
    }

    llama_set_layer_range(ctx, layer_start, layer_end);

    dist_debug_load_config();
    dist_debug_reset_recorder("entry");
    trace_recorder * dbg = dist_debug_recorder();
    int32_t debug_step   = 0;

    const int b_fd = split_tcp_connect_retry(b_host, b_port, 3000, 100);
    if (b_fd < 0) {
        split_gen_write_ready_state(ready_file, "FAILED");
        fprintf(stderr, "gen3_a: connect to B failed %s:%d\n", b_host, b_port);
        split_tcp_close(listen_fd);
        return 1;
    }
    split_gen_write_ready_state(ready_file, "PIPE_READY");

    fprintf(stderr, "gen3_a: ready ctrl_port=%d layer_start=%d layer_end=%d\n",
            ctrl_port, layer_start, layer_end);
    split_gen_write_ready_state(ready_file, "READY");

    bool normal_shutdown = false;
    bool pipe_failed = false;
    while (true) {
        const int ctrl_fd = split_tcp_accept(listen_fd);
        if (ctrl_fd < 0) {
            fprintf(stderr, "gen3_a: ctrl accept failed\n");
            continue;
        }

        int32_t entry_queue_depth_sync = 0;
        entry_stage_state st{
            ctx, b_fd, ctrl_fd, layer_start, layer_end, n_embd, n_vocab,
            next_is_final, dbg, &debug_step, &entry_queue_depth_sync
        };
        st.fa_buf = fa_port > 0 ? &fa_buf : nullptr;

        // A session with a draft wired up always takes the verify-wave
        // speculative path (see node_agent.cpp's matching
        // g_pipeline_fa_port > 0 check), which requires entry_queue off --
        // decided per session here instead of only via the process-wide
        // DIST_RUNTIME_ENTRY_QUEUE env var, since a session's fa_port is
        // known regardless of which node ended up holding entry.
        if (fa_port <= 0 && runtime_entry_queue_enabled()) {
            wave_inbound_queue queue(runtime_entry_queue_max_depth());
            if (!entry_run_queued_session(st, queue, debug_step, normal_shutdown, pipe_failed)) {
                split_tcp_close(ctrl_fd);
            } else {
                split_tcp_close(ctrl_fd);
            }
            if (normal_shutdown) {
                split_tcp_close(b_fd);
                split_tcp_close(listen_fd);
                llama_free(ctx);
                llama_model_free(model);
                split_gen_write_ready_state(ready_file, "STOPPED");
                return 0;
            }
            if (pipe_failed) {
                break;
            }
            continue;
        }

        while (true) {
        split_gen_a_req req{};
        std::vector<int32_t> tokens_i32;
        std::vector<float> hidden_in;
        int32_t hidden_n_embd = 0;
        perf_trace_sched_queue_wait_begin();
        const bool got_req = split_gen_recv_req(ctrl_fd, req, tokens_i32, &hidden_in, &hidden_n_embd);
        perf_trace_sched_queue_wait_end();
        if (!got_req) {
            fprintf(stderr, "gen3_a: client disconnected, waiting for next ctrl connection\n");
            split_tcp_close(ctrl_fd);
            break;
        }

        if (req.cmd == SPLIT_GEN_CMD_PROTO_NEGOTIATE ||
                req.cmd == SPLIT_GEN_CMD_SHUTDOWN ||
                req.cmd == SPLIT_GEN_CMD_RESET) {
            if (req.cmd == SPLIT_GEN_CMD_SHUTDOWN) {
                if (!entry_handle_control_cmd({
                        ctx, b_fd, ctrl_fd, layer_start, layer_end, n_embd, n_vocab,
                        next_is_final, dbg, &debug_step, &entry_queue_depth_sync
                    }, req, debug_step, normal_shutdown)) {
                    split_tcp_close(ctrl_fd);
                    break;
                }
                split_tcp_close(ctrl_fd);
                split_tcp_close(b_fd);
                split_tcp_close(listen_fd);
                llama_free(ctx);
                llama_model_free(model);
                split_gen_write_ready_state(ready_file, "STOPPED");
                return 0;
            }
            if (!entry_handle_control_cmd({
                    ctx, b_fd, ctrl_fd, layer_start, layer_end, n_embd, n_vocab,
                    next_is_final, dbg, &debug_step, &entry_queue_depth_sync
                }, req, debug_step, normal_shutdown)) {
                split_tcp_close(ctrl_fd);
                break;
            }
            continue;
        }

        wave_work_item item;
        item.req           = req;
        item.tokens        = std::move(tokens_i32);
        item.hidden        = std::move(hidden_in);
        item.hidden_n_embd = hidden_n_embd;
        const char * phase = (req.cmd == SPLIT_GEN_CMD_PREFILL ||
                              req.cmd == SPLIT_GEN_CMD_PREFILL_HIDDEN ||
                              req.cmd == SPLIT_GEN_CMD_VERIFY) ? "prefill" : "decode";
        item.wave_id = perf_trace_wave_id_from_step(phase, debug_step);

        if (!entry_process_work_item(st, item, false, nullptr, false, nullptr, nullptr)) {
            split_tcp_close(ctrl_fd);
            pipe_failed = true;
            break;
        }
        }
        if (pipe_failed) {
            break;
        }
    }

    if (!normal_shutdown) {
        split_gen_write_ready_state(ready_file, pipe_failed ? "FAILED" : "STOPPED");
    }
    split_tcp_close(b_fd);
    split_tcp_close(listen_fd);

    llama_free(ctx);
    llama_model_free(model);
    return 0;
}
