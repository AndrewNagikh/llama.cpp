// Process C: layers [layer_start, layer_end) or [layer_start, n_layer) - final stage

#include "ggml-backend.h"
#include "ggml.h"
#include "split_gen_model_load.h"
#include "split_gen_common.h"
#include "../transport/split_tcp_wire.h"
#include "../runtime_debug/debug_hooks.h"
#include "../runtime_debug/runtime_debug.h"
#include "../runtime_debug/trace_recorder.h"
#include "../runtime_debug/hidden_transport.h"
#include "../runtime_debug/perf_trace.h"
#include "../runtime_debug/perf_ggml.h"
#include "wave_inbound_queue.h"

#include "llama-distributed.h"

#include <httplib.h>
#include <nlohmann/json.hpp>

#include <atomic>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <map>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

using json = nlohmann::json;

// DIST_RUNTIME_SAMPLER_SYNC_SPLIT=1: explicit llama_synchronize before the
// sampler so the GPU wait is traced apart from the sampler chain. Default off.
static bool final_sampler_sync_split_enabled() {
    static const bool enabled = [] {
        const char * v = std::getenv("DIST_RUNTIME_SAMPLER_SYNC_SPLIT");
        if (v == nullptr || v[0] == '\0') {
            return false;
        }
        return std::strcmp(v, "0") != 0 &&
               std::strcmp(v, "false") != 0 &&
               std::strcmp(v, "FALSE") != 0;
    }();
    return enabled;
}

static void usage(const char * prog) {
    fprintf(stderr,
            "usage: %s MODEL --bc-port PORT [--layer-start N] [--layer-end N] "
            "[--external-output] [--output-http-host HOST] [--output-http-port PORT] "
            "[--ready-file PATH]\n",
            prog);
}

static bool output_service_sample_http(
        const std::string & http_host,
        int http_port,
        const float * hidden,
        int32_t n_embd,
        int32_t pos,
        int32_t & token_out,
        int32_t & n_vocab_out) {
    httplib::Client cli(http_host.c_str(), http_port);
    cli.set_connection_timeout(10, 0);
    cli.set_read_timeout(120, 0);
    json hidden_json = json::array();
    for (int32_t i = 0; i < n_embd; ++i) {
        hidden_json.push_back(hidden[i]);
    }
    const json body = {
        { "hidden", hidden_json },
        { "n_embd", n_embd },
        { "pos", pos },
    };
    const auto res = cli.Post("/runtime/output/sample", body.dump(), "application/json");
    if (!res || res->status != 200) {
        return false;
    }
    try {
        const json j = json::parse(res->body);
        if (!j.value("ok", false)) {
            return false;
        }
        token_out   = j.value("token_id", -1);
        n_vocab_out = j.value("n_vocab", 0);
        return token_out >= 0;
    } catch (...) {
        return false;
    }
}

static bool output_service_reset_http(const std::string & http_host, int http_port) {
    if (http_port <= 0) {
        return false;
    }
    httplib::Client cli(http_host.c_str(), http_port);
    cli.set_connection_timeout(10, 0);
    cli.set_read_timeout(60, 0);
    const auto res = cli.Post("/runtime/output/reset", "{}", "application/json");
    if (!res || res->status != 200) {
        return false;
    }
    try {
        const json j = json::parse(res->body);
        return j.value("ok", false);
    } catch (...) {
        return false;
    }
}

static bool run_partial_layers(
        llama_context * ctx,
        const split_tcp_hidden_msg & msg,
        int32_t n_embd,
        int layer_start,
        int layer_end,
        double & ms_out,
        std::vector<float> & out_hidden) {
    llama_set_layer_range(ctx, layer_start, layer_end);

    const int32_t n_tokens  = msg.header.n_tokens;
    const int32_t pos_start = msg.meta.pos_start;

    const int64_t t0 = ggml_time_us();

    if (n_tokens <= 1) {
        llama_set_hidden_state(ctx, msg.data.data(), n_tokens);
        {
            const hidden_state_api_check api = verify_hidden_state_roundtrip(
                    ctx, msg.data.data(), n_tokens, n_embd);
            if (!api.ok && dist_debug_transport_dump_enabled()) {
                fprintf(stderr, "gen3_c: hidden API roundtrip FAIL: %s\n", api.message.c_str());
            }
        }
        if (split_gen_decode_hidden(ctx, msg.data.data(), n_tokens, n_embd, pos_start, true) != 0) {
            fprintf(stderr, "gen3_c: decode hidden failed n_tokens=%d\n", n_tokens);
            return false;
        }
        const float * h = llama_get_embeddings(ctx);
        if (h == nullptr) {
            fprintf(stderr, "gen3_c: missing embeddings after decode n_tokens=%d\n", n_tokens);
            return false;
        }
        out_hidden.assign(h, h + (size_t) n_tokens * n_embd);
    } else {
        out_hidden.resize((size_t) n_tokens * n_embd);
        for (int32_t i = 0; i < n_tokens; ++i) {
            const float * in = msg.data.data() + (size_t) i * n_embd;
            llama_set_hidden_state(ctx, in, 1);
            {
                const hidden_state_api_check api = verify_hidden_state_roundtrip(ctx, in, 1, n_embd);
                if (!api.ok && dist_debug_transport_dump_enabled()) {
                    fprintf(stderr, "gen3_c: hidden API roundtrip FAIL i=%d: %s\n", i, api.message.c_str());
                }
            }
            if (split_gen_decode_hidden(ctx, in, 1, n_embd, pos_start + i, true) != 0) {
                fprintf(stderr, "gen3_c: decode hidden failed i=%d/%d\n", i, n_tokens);
                return false;
            }
            const float * h = llama_get_embeddings(ctx);
            if (h == nullptr) {
                fprintf(stderr, "gen3_c: missing embeddings i=%d/%d\n", i, n_tokens);
                return false;
            }
            std::memcpy(out_hidden.data() + (size_t) i * n_embd, h, (size_t) n_embd * sizeof(float));
        }
    }

    ms_out = (ggml_time_us() - t0) / 1000.0;
    return true;
}

struct final_stage_state {
    llama_context * ctx                 = nullptr;
    llama_model *   model               = nullptr;
    llama_sampler * smpl                = nullptr;
    int             bc_fd               = -1;
    int             layer_start         = 0;
    int             effective_layer_end = 0;
    int32_t         n_layer             = 0;
    int32_t         n_vocab             = 0;
    bool            external_output     = false;
    std::string     output_http_host;
    int             output_http_port    = 0;
    trace_recorder * dbg                = nullptr;
    int32_t *       debug_step          = nullptr;
    int32_t *       queue_depth         = nullptr;
    // Next expected KV position; a wave arriving below it means an earlier
    // speculative tail (or a retried wave) must be truncated first.
    int32_t         next_pos            = 0;
    // Task 19 Phase 3 draft placement (TASK_19_SPECULATIVE_PIPELINE_STUDY.md
    // SA): the draft model lives on the node holding `final`, so it can
    // guess the next wave locally while the current wave's result is still
    // in flight back to entry. Null draft_ctx = speculation disabled.
    llama_context * draft_ctx    = nullptr;
    int32_t         draft_n_vocab = 0;
    int32_t         draft_k       = 4;
    int32_t         draft_next_pos = 0;
    int             fa_fd         = -1;
    std::mutex *    draft_mu      = nullptr;
    // Set in the queued-session path: hands (anchor_tok, anchor_pos) to a
    // dedicated draft thread instead of running the k draft decodes inline
    // on the consumer thread, which would delay the next wave's decode by
    // the full draft compute time.
    std::function<void(int32_t, int32_t)> draft_submit;
    // Confirmed token ids by position (guarded by draft_mu). The draft's KV
    // must stay consecutive, but it can fall behind the target whenever a
    // ship is skipped or a wave fully accepts (the bonus token was never a
    // draft input); this journal lets final_draft_and_ship replay exactly
    // the confirmed tokens spanning any gap instead of dying on the first
    // non-consecutive decode.
    std::map<int32_t, int32_t> * spec_confirmed = nullptr;
};

static int32_t logits_argmax(const float * logits, const int32_t n_vocab) {
    int32_t best = 0;
    for (int32_t v = 1; v < n_vocab; ++v) {
        if (logits[v] > logits[best]) {
            best = v;
        }
    }
    return best;
}

// Verify wave: inputs are [anchor, d_1..d_k] at positions pos_start..pos_start+k.
// Decoding position pos_start+i predicts the token at pos_start+i+1; accept the
// longest prefix where the prediction matches d_{i+1}, stop at the first
// mismatch (the rejected tail is then never written to KV), and return the
// target's own prediction at the split point as the corrected/bonus token.
static bool final_verify_wave(
        final_stage_state & st,
        const split_tcp_hidden_msg & msg,
        const std::vector<int32_t> & ids,
        int32_t & accepted_out,
        int32_t & corrected_out,
        double & ms_compute_out) {
    const int32_t n_tokens = msg.header.n_tokens;
    const int32_t n_embd   = msg.header.n_embd;
    const int32_t pos_start = msg.meta.pos_start;
    if ((int32_t) ids.size() != n_tokens || n_tokens < 1) {
        fprintf(stderr, "gen3_c: verify ids/token count mismatch %zu vs %d\n", ids.size(), n_tokens);
        return false;
    }

    llama_set_layer_range(st.ctx, st.layer_start, st.n_layer);
    const int64_t t0 = ggml_time_us();

    // One batched decode with logits at every position: memory-bound, so
    // the whole wave costs about one token's compute instead of (k+1)x.
    // The rejected tail does get written to KV, but the standard rollback
    // rule already handles that -- the next wave arrives with pos_start
    // below next_pos and truncates it before decoding.
    llama_set_hidden_state(st.ctx, msg.data.data(), n_tokens);
    if (split_gen_decode_hidden(st.ctx, msg.data.data(), n_tokens, n_embd, pos_start, true) != 0) {
        fprintf(stderr, "gen3_c: verify batch decode failed n=%d\n", n_tokens);
        return false;
    }

    int32_t accepted  = 0;
    int32_t corrected = -1;
    for (int32_t i = 0; i < n_tokens; ++i) {
        const float * logits = llama_get_logits_ith(st.ctx, i);
        if (logits == nullptr) {
            fprintf(stderr, "gen3_c: verify logits missing i=%d\n", i);
            return false;
        }
        const int32_t pred = logits_argmax(logits, st.n_vocab);
        if (i < n_tokens - 1) {
            if (pred == ids[i + 1]) {
                accepted++;
                continue;
            }
            corrected = pred;
            break;
        }
        corrected = pred;
    }

    ms_compute_out = (ggml_time_us() - t0) / 1000.0;
    accepted_out   = accepted;
    corrected_out  = corrected;
    // KV holds the whole decoded batch [pos_start, pos_start + n_tokens);
    // the accepted prefix ends at pos_start + accepted and the corrected
    // token arrives as the next wave's anchor at pos_start + accepted + 1,
    // so that wave's rollback truncates the rejected tail.
    st.next_pos = pos_start + n_tokens;
    return true;
}

// Draft k tokens from (anchor_tok, anchor_pos) and ship them to entry over
// the fa-link. Runs synchronously right after final's response for the
// current wave has already been sent, so this compute overlaps with the
// token's return trip through middle/entry back to the client -- the
// "hidden under R" window from SA. Best-effort: any failure just leaves
// entry's next wave without a draft (falls back to plain decode there).
static void final_draft_and_ship(final_stage_state & st, int32_t anchor_tok, int32_t anchor_pos) {
    if (st.draft_ctx == nullptr || st.fa_fd < 0 || st.draft_mu == nullptr) {
        return;
    }
    std::lock_guard<std::mutex> lock(*st.draft_mu);
    // Replay any confirmed tokens the draft never ingested (skipped ships,
    // fully-accepted bonus tokens, late prime) so its KV stays consecutive
    // up to the anchor. Without this the first gap permanently killed the
    // draft: every later decode failed on non-consecutive positions.
    while (st.draft_next_pos < anchor_pos && st.spec_confirmed != nullptr) {
        const auto it = st.spec_confirmed->find(st.draft_next_pos);
        if (it == st.spec_confirmed->end()) {
            break;
        }
        if (split_gen_decode_one(st.draft_ctx, (llama_token) it->second, st.draft_next_pos) != 0) {
            fprintf(stderr, "gen3_c: draft catch-up decode failed\n");
            return;
        }
        st.draft_next_pos++;
    }
    split_gen_rollback_kv(st.draft_ctx, st.draft_next_pos, anchor_pos, "gen3_c_draft");
    if (st.draft_next_pos < anchor_pos) {
        fprintf(stderr, "gen3_c: draft skipped, kv gap [%d, %d)\n", st.draft_next_pos, anchor_pos);
        return;
    }
    if (st.spec_confirmed != nullptr) {
        st.spec_confirmed->erase(
                st.spec_confirmed->begin(), st.spec_confirmed->lower_bound(st.draft_next_pos));
    }

    std::vector<int32_t> ids;
    ids.reserve((size_t) st.draft_k);
    int32_t cur = anchor_tok;
    int32_t pos = anchor_pos;
    for (int32_t i = 0; i < st.draft_k; ++i) {
        if (split_gen_decode_one(st.draft_ctx, (llama_token) cur, pos) != 0) {
            fprintf(stderr, "gen3_c: draft decode failed\n");
            return;
        }
        const float * logits = llama_get_logits_ith(st.draft_ctx, -1);
        if (logits == nullptr) {
            fprintf(stderr, "gen3_c: draft logits missing\n");
            return;
        }
        cur = logits_argmax(logits, st.draft_n_vocab);
        pos++;
        ids.push_back(cur);
    }
    st.draft_next_pos = pos;
    if (!split_ab_send_verify_ids(st.fa_fd, anchor_pos, ids.data(), (int32_t) ids.size())) {
        fprintf(stderr, "gen3_c: ship draft ids failed\n");
    }
}

// Prime the draft context with the same prompt tokens the target pipeline
// just prefilled, so its continuations are grounded in the real context
// instead of just the bare anchor token. Entry ships these once per
// prefill over the fa-link (see split_gen3_a.cpp's PREFILL handling).
static bool final_draft_prefill(llama_context * draft_ctx, const std::vector<int32_t> & ids) {
    if (ids.empty()) {
        return true;
    }
    llama_memory_clear(llama_get_memory(draft_ctx), true);
    llama_batch batch = llama_batch_init((int32_t) ids.size(), 0, 1);
    for (size_t i = 0; i < ids.size(); ++i) {
        batch.token[i]     = (llama_token) ids[i];
        batch.pos[i]       = (llama_pos) i;
        batch.n_seq_id[i]  = 1;
        batch.seq_id[i][0] = 0;
        batch.logits[i]    = (i + 1 == ids.size());
    }
    batch.n_tokens = (int32_t) ids.size();
    const int rc = llama_decode(draft_ctx, batch);
    llama_batch_free(batch);
    return rc == 0;
}

static bool final_process_hidden_item(final_stage_state & st, hidden_wave_work_item & item) {
    split_tcp_hidden_msg & msg = item.msg;
    const char * phase = msg.header.n_tokens > 1 ? "prefill" : "decode";
    const int32_t step = item.debug_step;
    const int32_t wave_id = item.wave_id >= 0
            ? item.wave_id
            : perf_trace_wave_id_from_step(phase, step);
    const int32_t tok_idx = (std::strcmp(phase, "decode") == 0) ? step : -1;
    const bool decode_step = (std::strcmp(phase, "decode") == 0);

    if (decode_step && perf_trace_enabled()) {
        perf_trace_ensure_decode_context(tok_idx, wave_id);
        perf_trace_set_component("final");
        perf_emit_instant("FINAL_RECEIVE", perf_category::NETWORK, "final", tok_idx, nullptr);
        if (st.queue_depth) {
            perf_emit_queue_depth("final", tok_idx, *st.queue_depth);
            *st.queue_depth = std::max(0, *st.queue_depth - 1);
        }
    }

    hidden_transport_trace tr_recv = dist_debug_transport_recv(
            step, phase, "bc",
            msg.header.n_tokens, msg.header.n_embd, msg.header.layer_end,
            msg.meta.pos_start, msg.data.data(), 0.0);
    if (st.dbg) {
        st.dbg->emit_transport(tr_recv);
        st.dbg->emit_step_begin(step, phase, -1, msg.meta.pos_start, 0);
        dist_debug_log_position(st.dbg, step, phase, st.ctx, msg.meta.pos_start,
                msg.header.n_tokens, 1);
    }
    if (!tr_recv.memcmp_ok && dist_debug_transport_dump_enabled()) {
        fprintf(stderr, "gen3_c: TCP hidden memcmp FAIL step=%d link=bc\n", step);
    }

    const int32_t n_emb = msg.header.n_embd > 0 ? msg.header.n_embd : llama_model_n_embd(st.model);
    const int32_t n_tok = msg.header.n_tokens > 0 ? msg.header.n_tokens : 1;
    const int32_t sample_idx = n_tok - 1;
    const int32_t sample_pos = msg.meta.pos_start + sample_idx;

    double ms_compute = 0.0;
    int32_t token_id = -1;
    int32_t resp_vocab = st.n_vocab;
    double ms_sample = 0.0;

    split_gen_rollback_kv(st.ctx, st.next_pos, msg.meta.pos_start, "gen3_c");

    if (!item.verify_ids.empty()) {
        if (st.external_output) {
            fprintf(stderr, "gen3_c: verify waves unsupported with external output\n");
            return false;
        }
        int32_t accepted = 0;
        if (!final_verify_wave(st, msg, item.verify_ids, accepted, token_id, ms_compute)) {
            return false;
        }
        if (st.dbg) {
            st.dbg->emit_token_selected(step, phase, token_id, msg.meta.pos_start, true);
        }
        split_gen3_c_resp resp{};
        resp.magic          = SPLIT_GEN_MAGIC;
        resp.token_id       = token_id;
        resp.n_vocab        = resp_vocab;
        resp.ms_compute     = ms_compute;
        resp.ms_sample      = 0.0;
        resp.accepted_count = accepted;
        if (!split_gen3_send_c_resp(st.bc_fd, resp)) {
            fprintf(stderr, "gen3_c: send verify resp failed\n");
            return false;
        }
        // st.next_pos now covers the whole decoded batch (rollback anchor);
        // the corrected token itself lives right after the accepted prefix.
        const int32_t corrected_pos = msg.meta.pos_start + accepted + 1;
        if (st.spec_confirmed != nullptr && st.draft_mu != nullptr) {
            std::lock_guard<std::mutex> lock(*st.draft_mu);
            for (int32_t i = 0; i <= accepted && i < (int32_t) item.verify_ids.size(); ++i) {
                (*st.spec_confirmed)[msg.meta.pos_start + i] = item.verify_ids[(size_t) i];
            }
            (*st.spec_confirmed)[corrected_pos] = token_id;
        }
        if (st.draft_submit) {
            st.draft_submit(token_id, corrected_pos);
        } else {
            final_draft_and_ship(st, token_id, corrected_pos);
        }
        if (st.debug_step) {
            (*st.debug_step)++;
        }
        return true;
    }

    if (st.external_output) {
        std::vector<float> out_hidden;
        split_gen_pipe_trace("final", phase, "partial_decode_enter",
                msg.header.n_tokens, n_emb, msg.meta.pos_start, st.layer_start, st.effective_layer_end);
        if (!run_partial_layers(
                    st.ctx, msg, n_emb, st.layer_start, st.effective_layer_end, ms_compute, out_hidden)) {
            fprintf(stderr, "gen3_c: partial forward failed\n");
            return false;
        }
        split_gen_pipe_trace("final", phase, "partial_decode_exit",
                msg.header.n_tokens, n_emb, msg.meta.pos_start, st.layer_start, st.effective_layer_end);
        const float * sample_hidden = out_hidden.data() + (size_t) sample_idx * (size_t) n_emb;
        const int64_t t1 = ggml_time_us();
        perf_span sample_span("SAMPLER_BEGIN", "SAMPLER_END", perf_category::SAMPLING, "final");
        sample_span.set_token_idx(tok_idx);
        if (st.output_http_port <= 0 ||
                !output_service_sample_http(
                        st.output_http_host, st.output_http_port,
                        sample_hidden, n_emb, sample_pos, token_id, resp_vocab)) {
            fprintf(stderr, "gen3_c: output service sample failed\n");
            return false;
        }
        ms_sample = (ggml_time_us() - t1) / 1000.0;
    } else {
        llama_set_layer_range(st.ctx, st.layer_start, st.n_layer);
        llama_set_hidden_state(st.ctx, msg.data.data(), msg.header.n_tokens);
        {
            const hidden_state_api_check api = verify_hidden_state_roundtrip(
                    st.ctx, msg.data.data(), msg.header.n_tokens, msg.header.n_embd);
            if (!api.ok && dist_debug_transport_dump_enabled()) {
                fprintf(stderr, "gen3_c: hidden API roundtrip FAIL: %s\n", api.message.c_str());
            }
        }

        const int64_t t0 = ggml_time_us();
        split_gen_pipe_trace("final", phase, "decode_enter",
                msg.header.n_tokens, msg.header.n_embd, msg.meta.pos_start, st.layer_start, st.n_layer);
        if (perf_trace_enabled() && decode_step) {
            perf_trace_ensure_decode_context(tok_idx, wave_id);
        }
        perf_span compute_span("FINAL_COMPUTE_BEGIN", "FINAL_COMPUTE_END", perf_category::COMPUTE, "final");
        compute_span.set_token_idx(tok_idx);
        if (split_gen_decode_hidden(st.ctx, msg.data.data(), msg.header.n_tokens, msg.header.n_embd,
                    msg.meta.pos_start, true) != 0) {
            fprintf(stderr, "gen3_c: decode failed\n");
            return false;
        }
        split_gen_pipe_trace("final", phase, "decode_exit",
                msg.header.n_tokens, msg.header.n_embd, msg.meta.pos_start, st.layer_start, st.n_layer);
        ms_compute = (ggml_time_us() - t0) / 1000.0;

        if (st.dbg) {
            dist_debug_log_kv(st.dbg, step, phase, st.ctx, 0);
            dist_debug_log_hidden_out(st.dbg, step, phase, st.ctx, msg.header.n_tokens, n_emb, "final");
            dist_debug_log_logits_out(st.dbg, step, phase, st.ctx, st.n_vocab, -1);
        }

        const int64_t t1 = ggml_time_us();
        perf_span sample_span("SAMPLER_BEGIN", "SAMPLER_END", perf_category::SAMPLING, "final");
        sample_span.set_token_idx(tok_idx);
        bool used_argmax = false;
        const llama_token sampled = static_cast<llama_token>(
                dist_debug_sample_or_argmax(st.smpl, st.ctx, -1, &used_argmax));
        if (!used_argmax) {
            llama_sampler_accept(st.smpl, sampled);
        }
        token_id = (int32_t) sampled;
        ms_sample = (ggml_time_us() - t1) / 1000.0;

        if (st.dbg) {
            st.dbg->emit_token_selected(step, phase, token_id, msg.meta.pos_start, used_argmax);
        }
    }

    if (st.dbg) {
        dist_debug_log_runtime_state(
                st.dbg, step, phase, "final", st.ctx, msg.header.n_tokens,
                nullptr, nullptr, -1, token_id);
    }

    st.next_pos = msg.meta.pos_start + n_tok;

    split_gen3_c_resp resp{};
    resp.magic          = SPLIT_GEN_MAGIC;
    resp.token_id       = token_id;
    resp.n_vocab        = resp_vocab;
    resp.ms_compute     = ms_compute;
    resp.ms_sample      = ms_sample;
    resp.accepted_count = -1;

    if (!split_gen3_send_c_resp(st.bc_fd, resp)) {
        fprintf(stderr, "gen3_c: send resp failed\n");
        return false;
    }
    if (st.spec_confirmed != nullptr && st.draft_mu != nullptr) {
        std::lock_guard<std::mutex> lock(*st.draft_mu);
        (*st.spec_confirmed)[st.next_pos] = token_id;
    }
    if (st.draft_submit) {
        st.draft_submit(token_id, st.next_pos);
    } else {
        final_draft_and_ship(st, token_id, st.next_pos);
    }
    if (decode_step && st.queue_depth) {
        (*st.queue_depth)++;
    }
    if (st.debug_step) {
        (*st.debug_step)++;
    }
    return true;
}

static bool final_run_queued_session(
        final_stage_state st,
        hidden_inbound_queue & queue,
        int32_t & debug_step,
        bool & normal_shutdown,
        bool & pipe_failed) {
    std::atomic<bool> stop{ false };
    int32_t final_queue_depth = 0;
    st.debug_step = &debug_step;
    st.queue_depth = &final_queue_depth;

    // RESET arrives on the receiver thread's control-channel recv loop, while
    // the consumer loop below concurrently drives decode/sample on the same
    // st.ctx from queued hidden-state items. Without this lock the two
    // threads mutate the same llama_context (KV cache clear vs. active
    // decode) at once, which corrupts ggml's allocator state -- seen as
    // "malloc: Heap corruption detected" crashes on the final-stage worker.
    std::mutex ctx_mu;

    // Draft compute runs on its own thread so the consumer loop can start
    // decoding the next wave immediately after sending a response instead
    // of stalling for the k draft decode steps. Single-slot mailbox: only
    // the latest (anchor, pos) matters -- a newer wave's result obsoletes
    // any draft not yet started.
    std::mutex              draft_q_mu;
    std::condition_variable draft_q_cv;
    int32_t                 draft_q_tok  = -1;
    int32_t                 draft_q_pos  = -1;
    bool                    draft_q_stop = false;
    std::thread             draft_thread;
    std::map<int32_t, int32_t> spec_confirmed;
    if (st.draft_ctx != nullptr && st.fa_fd >= 0) {
        st.spec_confirmed = &spec_confirmed;
        st.draft_submit = [&](int32_t tok, int32_t pos) {
            {
                std::lock_guard<std::mutex> lk(draft_q_mu);
                draft_q_tok = tok;
                draft_q_pos = pos;
            }
            draft_q_cv.notify_one();
        };
        draft_thread = std::thread([&] {
            while (true) {
                int32_t tok, pos;
                {
                    std::unique_lock<std::mutex> lk(draft_q_mu);
                    draft_q_cv.wait(lk, [&] { return draft_q_stop || draft_q_pos >= 0; });
                    if (draft_q_stop) {
                        break;
                    }
                    tok = draft_q_tok;
                    pos = draft_q_pos;
                    draft_q_tok = -1;
                    draft_q_pos = -1;
                }
                final_draft_and_ship(st, tok, pos);
            }
        });
    }

    std::atomic<bool> fa_stop{ false };
    std::thread fa_prime;
    if (st.draft_ctx != nullptr && st.fa_fd >= 0) {
        fa_prime = std::thread([&] {
            while (!fa_stop.load()) {
                split_ab_cmd cmd;
                if (!split_ab_recv_cmd(st.fa_fd, cmd) || cmd != SPLIT_AB_CMD_VERIFY_IDS) {
                    fprintf(stderr, "SCRATCH_DEBUG final: fa_prime recv_cmd fail or wrong cmd\n");
                    break;
                }
                int32_t pos = -1;
                std::vector<int32_t> ids;
                if (!split_ab_recv_verify_ids(st.fa_fd, pos, ids)) {
                    fprintf(stderr, "SCRATCH_DEBUG final: fa_prime recv_ids fail\n");
                    break;
                }
                fprintf(stderr, "SCRATCH_DEBUG final: fa_prime got pos=%d n=%zu\n", pos, ids.size());
                std::lock_guard<std::mutex> lock(*st.draft_mu);
                if (!final_draft_prefill(st.draft_ctx, ids)) {
                    fprintf(stderr, "gen3_c: draft prefill failed\n");
                    continue;
                }
                st.draft_next_pos = (int32_t) ids.size();
                if (st.spec_confirmed != nullptr) {
                    st.spec_confirmed->clear();
                }
                fprintf(stderr, "SCRATCH_DEBUG final: primed D=%d\n", st.draft_next_pos);
            }
        });
    }

    std::thread receiver([&] {
        int32_t              pending_verify_pos = -1;
        std::vector<int32_t> pending_verify_ids;
        while (!stop.load()) {
            split_ab_cmd cmd;
            perf_trace_sched_queue_wait_begin();
            const bool got_cmd = split_ab_recv_cmd(st.bc_fd, cmd);
            perf_trace_sched_queue_wait_end();
            if (!got_cmd) {
                stop.store(true);
                break;
            }

            if (cmd == SPLIT_AB_CMD_SHUTDOWN) {
                normal_shutdown = true;
                stop.store(true);
                break;
            }

            if (cmd == SPLIT_AB_CMD_RESET) {
                {
                    std::lock_guard<std::mutex> lock(ctx_mu);
                    llama_memory_clear(llama_get_memory(st.ctx), true);
                    llama_clear_hidden_state(st.ctx);
                }
                // Deliberately no draft-state reset here: the prime for the
                // next session arrives over the direct fa-link and often
                // beats this RESET (which crawls the bc chain) -- clearing
                // draft_next_pos now would wipe an already-applied prime.
                // final_draft_prefill does its own memory_clear + reprime,
                // so the prime alone fully re-initializes the draft.
                if (st.external_output &&
                        !output_service_reset_http(st.output_http_host, st.output_http_port)) {
                    fprintf(stderr, "gen3_c: output service reset failed\n");
                    stop.store(true);
                    break;
                }
                debug_step = 0;
                final_queue_depth = 0;
                pending_verify_pos = -1;
                pending_verify_ids.clear();
                continue;
            }

            if (cmd == SPLIT_AB_CMD_VERIFY_IDS) {
                if (!split_ab_recv_verify_ids(st.bc_fd, pending_verify_pos, pending_verify_ids)) {
                    stop.store(true);
                    break;
                }
                continue;
            }

            if (cmd != SPLIT_AB_CMD_HIDDEN) {
                fprintf(stderr, "gen3_c: unexpected cmd %u\n", cmd);
                stop.store(true);
                break;
            }

            split_tcp_hidden_msg msg;
            perf_trace_sched_queue_wait_begin();
            const bool got_hidden = split_ab_recv_hidden(st.bc_fd, msg);
            perf_trace_sched_queue_wait_end();
            if (!got_hidden) {
                stop.store(true);
                break;
            }

            const char * phase = msg.header.n_tokens > 1 ? "prefill" : "decode";
            const int32_t queued_wave_id = perf_trace_wave_id_from_step(phase, debug_step);
            hidden_wave_work_item item;
            item.msg        = std::move(msg);
            item.debug_step = debug_step;
            item.wave_id    = queued_wave_id;
            if (pending_verify_pos >= 0 && pending_verify_pos == item.msg.meta.pos_start) {
                item.verify_ids = std::move(pending_verify_ids);
            }
            pending_verify_pos = -1;
            pending_verify_ids.clear();

            queue.push(std::move(item));
            final_queue_depth = queue.observable_depth(true);
            if (perf_trace_enabled()) {
                perf_trace_set_wave_id(queued_wave_id);
                perf_emit_instant("WAVE_QUEUED", perf_category::WAIT, "final", -1, nullptr);
                perf_emit_queue_depth("final", -1, final_queue_depth);
            }
        }
        // Wake a consumer parked in queue.pop(); nothing else ever will
        // once this producer thread has exited (see split_gen3_a.cpp).
        queue.close();
    });

    while (!stop.load() || queue.depth() > 0) {
        hidden_wave_work_item item;
        if (!queue.try_pop(item)) {
            if (stop.load()) {
                break;
            }
            if (!queue.pop(item)) {
                // Queue closed by the receiver and fully drained.
                break;
            }
        }
        bool ok;
        {
            std::lock_guard<std::mutex> lock(ctx_mu);
            ok = final_process_hidden_item(st, item);
        }
        if (!ok) {
            pipe_failed = true;
            stop.store(true);
            break;
        }
    }

    stop.store(true);
    // Wake a receiver parked in a blocking push() so the join can complete.
    queue.close();
    if (receiver.joinable()) {
        receiver.join();
    }
    if (draft_thread.joinable()) {
        {
            std::lock_guard<std::mutex> lk(draft_q_mu);
            draft_q_stop = true;
        }
        draft_q_cv.notify_all();
        draft_thread.join();
    }
    if (fa_prime.joinable()) {
        fa_stop.store(true);
        split_tcp_close(st.fa_fd);
        fa_prime.join();
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
    int bc_port     = -1;
    int layer_start = 8;
    int layer_end   = 0;
    std::string output_http_host = "127.0.0.1";
    int output_http_port = 0;
    bool external_output = false;
    const char * bind_host = "0.0.0.0";
    std::string ready_file;
    const char * draft_model_path = nullptr;
    const char * fa_host = "127.0.0.1";
    int fa_port = -1;
    int draft_k = 4;

    for (int i = 2; i < argc; ++i) {
        if (strcmp(argv[i], "--bc-port") == 0 && i + 1 < argc) {
            bc_port = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--bind") == 0 && i + 1 < argc) {
            bind_host = argv[++i];
        } else if (strcmp(argv[i], "--layer-start") == 0 && i + 1 < argc) {
            layer_start = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--layer-end") == 0 && i + 1 < argc) {
            layer_end = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--external-output") == 0) {
            external_output = true;
        } else if (strcmp(argv[i], "--output-http-host") == 0 && i + 1 < argc) {
            output_http_host = argv[++i];
        } else if (strcmp(argv[i], "--output-http-port") == 0 && i + 1 < argc) {
            output_http_port = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--ready-file") == 0 && i + 1 < argc) {
            ready_file = argv[++i];
        } else if (strcmp(argv[i], "--draft-model") == 0 && i + 1 < argc) {
            draft_model_path = argv[++i];
        } else if (strcmp(argv[i], "--fa-host") == 0 && i + 1 < argc) {
            fa_host = argv[++i];
        } else if (strcmp(argv[i], "--fa-port") == 0 && i + 1 < argc) {
            fa_port = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--draft-k") == 0 && i + 1 < argc) {
            draft_k = atoi(argv[++i]);
        } else {
            usage(argv[0]);
            return 1;
        }
    }

    if (bc_port < 0) {
        usage(argv[0]);
        return 1;
    }

    split_gen_write_ready_state(ready_file, "MODEL_LOADING");

    const int listen_fd = split_tcp_listen_host(bind_host, bc_port);
    if (listen_fd < 0) {
        split_gen_write_ready_state(ready_file, "FAILED");
        fprintf(stderr, "gen3_c: listen failed %s:%d\n", bind_host, bc_port);
        return 1;
    }
    split_gen_write_ready_state(ready_file, "LISTENER_READY");

    ggml_backend_load_all();

    std::string load_err;
    llama_model * model = split_gen_load_model(model_path, load_err);
    if (!model) {
        split_gen_write_ready_state(ready_file, "FAILED");
        fprintf(stderr, "gen3_c: load model failed: %s\n", load_err.c_str());
        return 1;
    }

    const int32_t n_layer = llama_model_n_layer(model);
    const int32_t n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model));
    int32_t effective_layer_end = layer_end > layer_start ? layer_end : n_layer;
    if (external_output && effective_layer_end >= n_layer && layer_start + 1 < n_layer) {
        effective_layer_end = n_layer - 1;
    }

    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx   = 512;
    cparams.n_batch = 512;
    // n_ubatch > 1 so a k+1-token verify wave runs as ONE graph compute
    // (memory-bound, ~one token's cost) instead of k+1 sequential
    // single-token graphs. Hidden injection is ubatch-agnostic: the whole
    // wave's hidden becomes batch.embd and the normal ubatch split applies
    // (see llama-context.cpp layer_start handling).
    cparams.n_ubatch = 32;
    cparams.no_perf = true;
    cparams.layer_start = layer_start;
    cparams.layer_end   = effective_layer_end;
    cparams.skip_output_head = external_output;
    fprintf(stderr, "gen3_c: create context layer_start=%d layer_end=%d skip_output_head=%d\n",
            layer_start, effective_layer_end, external_output ? 1 : 0);
    llama_context * ctx = llama_init_from_model(model, cparams);
    if (!ctx) {
        split_gen_write_ready_state(ready_file, "FAILED");
        fprintf(stderr, "gen3_c: create context failed\n");
        return 1;
    }

    if (external_output) {
        llama_set_skip_output_head(ctx, true);
    }

    llama_set_layer_range(ctx, layer_start, effective_layer_end);
    llama_sampler * smpl = external_output ? nullptr : split_gen_make_sampler();

    dist_debug_load_config();
    dist_debug_reset_recorder("final");
    trace_recorder * dbg = dist_debug_recorder();
    int32_t debug_step   = 0;

    // Task 19 Phase 3: draft model colocated with `final` (placement per
    // TASK_19_SPECULATIVE_PIPELINE_STUDY.md SA). Optional: only set up when
    // the orchestrator gave us a draft model to load.
    llama_model *   draft_model = nullptr;
    llama_context * draft_ctx   = nullptr;
    int             fa_fd       = -1;
    std::mutex      draft_mu;
    if (draft_model_path != nullptr) {
        // Not split_gen_load_model(): in layer-store mode (DIST_RUNTIME_LAYER_FIRST
        // + DIST_MODEL_ID set process-wide for this pipeline-stage worker) that
        // function ignores its model_path argument entirely and reloads whatever
        // DIST_MODEL_ID/DIST_WORKER_LAYER_START/_END point to -- i.e. it would
        // silently reload the *primary* target model a second time instead of the
        // draft, using the primary's layer range. The draft is always a whole,
        // unsliced file on disk; load it directly.
        std::string draft_load_err;
        draft_model = llama_model_load_from_file(draft_model_path, llama_model_default_params());
        if (!draft_model) {
            draft_load_err = "llama_model_load_from_file failed";
            fprintf(stderr, "gen3_c: draft model load failed: %s (speculation disabled)\n",
                    draft_load_err.c_str());
        } else {
            llama_context_params dcparams = llama_context_default_params();
            dcparams.n_ctx    = 2048;
            dcparams.n_batch  = 512;
            dcparams.no_perf  = true;
            draft_ctx = llama_init_from_model(draft_model, dcparams);
            if (!draft_ctx) {
                fprintf(stderr, "gen3_c: draft context create failed (speculation disabled)\n");
                llama_model_free(draft_model);
                draft_model = nullptr;
            } else if (fa_port > 0) {
                fa_fd = split_tcp_connect_retry(fa_host, fa_port, 300, 100);
                if (fa_fd < 0) {
                    fprintf(stderr, "gen3_c: fa connect failed %s:%d (speculation disabled)\n",
                            fa_host, fa_port);
                    llama_free(draft_ctx);
                    llama_model_free(draft_model);
                    draft_ctx = nullptr;
                    draft_model = nullptr;
                }
            }
        }
    }

    fprintf(stderr,
            "gen3_c: waiting bc_port=%d layer_start=%d layer_end=%d external_output=%d draft=%d\n",
            bc_port, layer_start, effective_layer_end, external_output ? 1 : 0, draft_ctx != nullptr ? 1 : 0);
    split_gen_write_ready_state(ready_file, "PIPE_READY");
    split_gen_write_ready_state(ready_file, "READY");

    const int bc_fd = split_tcp_accept(listen_fd);
    split_tcp_close(listen_fd);
    if (bc_fd < 0) {
        fprintf(stderr, "gen3_c: accept failed\n");
        split_gen_write_ready_state(ready_file, "FAILED");
        return 1;
    }

    bool normal_shutdown = false;
    bool pipe_failed = false;
    int32_t final_queue_depth = 0;

    if (runtime_stage_queue_enabled()) {
        hidden_inbound_queue queue(runtime_stage_queue_max_depth());
        final_stage_state st{
            ctx, model, smpl, bc_fd, layer_start, effective_layer_end, n_layer, n_vocab,
            external_output, output_http_host, output_http_port, dbg, &debug_step, &final_queue_depth
        };
        if (draft_ctx != nullptr) {
            st.draft_ctx     = draft_ctx;
            st.draft_n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(draft_model));
            st.draft_k       = draft_k;
            st.fa_fd         = fa_fd;
            st.draft_mu      = &draft_mu;
        }
        if (!final_run_queued_session(st, queue, debug_step, normal_shutdown, pipe_failed)) {
            split_gen_write_ready_state(ready_file, "FAILED");
        } else {
            split_gen_write_ready_state(ready_file, normal_shutdown ? "STOPPED" : "FAILED");
        }
        split_tcp_close(bc_fd);
        if (smpl) {
            llama_sampler_free(smpl);
        }
        if (draft_ctx) {
            llama_free(draft_ctx);
            llama_model_free(draft_model);
        }
        llama_free(ctx);
        llama_model_free(model);
        return pipe_failed ? 1 : 0;
    }

    while (true) {
        split_ab_cmd cmd;
        perf_trace_sched_queue_wait_begin();
        const bool got_cmd = split_ab_recv_cmd(bc_fd, cmd);
        perf_trace_sched_queue_wait_end();
        if (!got_cmd) {
            fprintf(stderr, "gen3_c: recv cmd failed\n");
            break;
        }

        if (cmd == SPLIT_AB_CMD_SHUTDOWN) {
            normal_shutdown = true;
            break;
        }

        if (cmd == SPLIT_AB_CMD_RESET) {
            llama_memory_clear(llama_get_memory(ctx), true);
            llama_clear_hidden_state(ctx);
            if (external_output && !output_service_reset_http(output_http_host, output_http_port)) {
                fprintf(stderr, "gen3_c: output service reset failed\n");
                break;
            }
            debug_step = 0;
            continue;
        }

        if (cmd == SPLIT_AB_CMD_VERIFY_IDS) {
            // Verify waves are only routed through the queued-session path.
            fprintf(stderr, "gen3_c: verify wave requires the stage queue path\n");
            break;
        }

        if (cmd != SPLIT_AB_CMD_HIDDEN) {
            fprintf(stderr, "gen3_c: unexpected cmd %u\n", cmd);
            break;
        }

        split_tcp_hidden_msg msg;
        perf_trace_sched_queue_wait_begin();
        const bool got_hidden = split_ab_recv_hidden(bc_fd, msg);
        perf_trace_sched_queue_wait_end();
        if (!got_hidden) {
            fprintf(stderr, "gen3_c: recv hidden failed\n");
            break;
        }

        const char * phase = msg.header.n_tokens > 1 ? "prefill" : "decode";
        const int32_t tok_idx = (std::strcmp(phase, "decode") == 0) ? debug_step : -1;
        const bool decode_step = (std::strcmp(phase, "decode") == 0);
        if (decode_step && perf_trace_enabled()) {
            perf_trace_ensure_decode_context(tok_idx, perf_trace_wave_id_from_step(phase, debug_step));
            perf_trace_set_component("final");
            perf_emit_instant("FINAL_RECEIVE", perf_category::NETWORK, "final", tok_idx, nullptr);
            perf_emit_queue_depth("final", tok_idx, final_queue_depth);
        }
        if (decode_step) {
            final_queue_depth = std::max(0, final_queue_depth - 1);
        }

        hidden_transport_trace tr_recv = dist_debug_transport_recv(
                debug_step, phase, "bc",
                msg.header.n_tokens, msg.header.n_embd, msg.header.layer_end,
                msg.meta.pos_start, msg.data.data(), 0.0);
        if (dbg) {
            dbg->emit_transport(tr_recv);
        }
        if (!tr_recv.memcmp_ok && dist_debug_transport_dump_enabled()) {
            fprintf(stderr, "gen3_c: TCP hidden memcmp FAIL step=%d link=bc\n", debug_step);
        }

        if (dbg) {
            dbg->emit_step_begin(debug_step, phase, -1, msg.meta.pos_start, 0);
            dist_debug_log_position(dbg, debug_step, phase, ctx, msg.meta.pos_start,
                    msg.header.n_tokens, 1);
        }

        const int32_t n_emb = msg.header.n_embd > 0 ? msg.header.n_embd : llama_model_n_embd(model);
        const int32_t n_tok = msg.header.n_tokens > 0 ? msg.header.n_tokens : 1;
        const int32_t sample_idx = n_tok - 1;
        const int32_t sample_pos = msg.meta.pos_start + sample_idx;

        double ms_compute = 0.0;
        int32_t token_id = -1;
        int32_t resp_vocab = n_vocab;
        double ms_sample = 0.0;

        if (external_output) {
            std::vector<float> out_hidden;
            split_gen_pipe_trace("final", phase, "partial_decode_enter",
                    msg.header.n_tokens, n_emb, msg.meta.pos_start, layer_start, effective_layer_end);
            if (!run_partial_layers(
                        ctx, msg, n_emb, layer_start, effective_layer_end, ms_compute, out_hidden)) {
                fprintf(stderr, "gen3_c: partial forward failed\n");
                break;
            }
            split_gen_pipe_trace("final", phase, "partial_decode_exit",
                    msg.header.n_tokens, n_emb, msg.meta.pos_start, layer_start, effective_layer_end);
            const float * sample_hidden = out_hidden.data() + (size_t) sample_idx * (size_t) n_emb;
            const int64_t t1 = ggml_time_us();
            perf_span sample_span("SAMPLER_BEGIN", "SAMPLER_END", perf_category::SAMPLING, "final");
            sample_span.set_token_idx(tok_idx);
            if (output_http_port <= 0 ||
                    !output_service_sample_http(
                            output_http_host, output_http_port,
                            sample_hidden, n_emb, sample_pos, token_id, resp_vocab)) {
                fprintf(stderr, "gen3_c: output service sample failed\n");
                break;
            }
            ms_sample = (ggml_time_us() - t1) / 1000.0;
        } else {
            llama_set_layer_range(ctx, layer_start, n_layer);
            llama_set_hidden_state(ctx, msg.data.data(), msg.header.n_tokens);
            {
                const hidden_state_api_check api = verify_hidden_state_roundtrip(
                        ctx, msg.data.data(), msg.header.n_tokens, msg.header.n_embd);
                if (!api.ok && dist_debug_transport_dump_enabled()) {
                    fprintf(stderr, "gen3_c: hidden API roundtrip FAIL: %s\n", api.message.c_str());
                }
            }

            const int64_t t0 = ggml_time_us();
            split_gen_pipe_trace("final", phase, "decode_enter",
                    msg.header.n_tokens, msg.header.n_embd, msg.meta.pos_start, layer_start, n_layer);
            if (perf_trace_enabled() && decode_step) {
                perf_trace_ensure_decode_context(
                        tok_idx, perf_trace_wave_id_from_step(phase, debug_step));
            }
            perf_span compute_span("FINAL_COMPUTE_BEGIN", "FINAL_COMPUTE_END", perf_category::COMPUTE, "final");
            compute_span.set_token_idx(tok_idx);
            if (split_gen_decode_hidden(ctx, msg.data.data(), msg.header.n_tokens, msg.header.n_embd,
                        msg.meta.pos_start, true) != 0) {
                fprintf(stderr, "gen3_c: decode failed\n");
                break;
            }
            split_gen_pipe_trace("final", phase, "decode_exit",
                    msg.header.n_tokens, msg.header.n_embd, msg.meta.pos_start, layer_start, n_layer);
            ms_compute = (ggml_time_us() - t0) / 1000.0;

            if (dbg) {
                dist_debug_log_kv(dbg, debug_step, phase, ctx, 0);
                dist_debug_log_hidden_out(dbg, debug_step, phase, ctx, msg.header.n_tokens, n_emb, "final");
                dist_debug_log_logits_out(dbg, debug_step, phase, ctx, n_vocab, -1);
            }

            const int64_t t1 = ggml_time_us();
            if (final_sampler_sync_split_enabled()) {
                // Isolate the GPU wait from the sampler chain: llama_get_logits*
                // inside the sampler would otherwise pay this synchronize.
                perf_span sync_span("FINAL_LOGITS_SYNC_BEGIN", "FINAL_LOGITS_SYNC_END", perf_category::GPU, "final");
                sync_span.set_token_idx(tok_idx);
                llama_synchronize(ctx);
            }
            perf_span sample_span("SAMPLER_BEGIN", "SAMPLER_END", perf_category::SAMPLING, "final");
            sample_span.set_token_idx(tok_idx);
            bool used_argmax = false;
            const llama_token sampled = static_cast<llama_token>(
                    dist_debug_sample_or_argmax(smpl, ctx, -1, &used_argmax));
            if (!used_argmax) {
                llama_sampler_accept(smpl, sampled);
            }
            token_id = (int32_t) sampled;
            ms_sample = (ggml_time_us() - t1) / 1000.0;

            if (dbg) {
                dbg->emit_token_selected(debug_step, phase, token_id, msg.meta.pos_start, used_argmax);
            }
        }

        if (dbg) {
            dist_debug_log_runtime_state(
                    dbg, debug_step, phase, "final", ctx, msg.header.n_tokens,
                    nullptr, nullptr, -1, token_id);
        }

        split_gen3_c_resp resp{};
        resp.magic          = SPLIT_GEN_MAGIC;
        resp.token_id       = token_id;
        resp.n_vocab        = resp_vocab;
        resp.ms_compute     = ms_compute;
        resp.ms_sample      = ms_sample;
        resp.accepted_count = -1;

        {
            perf_span resp_span("FINAL_RESP_SEND_BEGIN", "FINAL_RESP_SEND_END", perf_category::NETWORK, "final");
            resp_span.set_token_idx(tok_idx);
            if (!split_gen3_send_c_resp(bc_fd, resp)) {
                fprintf(stderr, "gen3_c: send resp failed\n");
                break;
            }
        }
        if (decode_step) {
            final_queue_depth++;
        }
        debug_step++;
    }

    split_gen_write_ready_state(ready_file, normal_shutdown ? "STOPPED" : "FAILED");
    split_tcp_close(bc_fd);

    if (smpl) {
        llama_sampler_free(smpl);
    }
    llama_free(ctx);
    llama_model_free(model);
    return 0;
}
