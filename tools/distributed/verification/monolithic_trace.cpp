#include "monolithic_trace.h"

#include "runtime_debug/debug_hooks.h"
#include "runtime_debug/runtime_debug.h"
#include "split_gen_common.h"

#include "ggml-backend.h"

#include <fstream>
#include <sstream>

namespace {

static int32_t argmax_logits(const float * logits, const int32_t n_vocab) {
    if (logits == nullptr || n_vocab <= 0) {
        return -1;
    }
    int32_t best = 0;
    for (int32_t i = 1; i < n_vocab; ++i) {
        if (logits[i] > logits[best]) {
            best = i;
        }
    }
    return best;
}

} // namespace

mono_trace_result run_monolithic_trace(
        const std::string & model_path,
        const std::string & prompt,
        const int max_tokens,
        const std::string & session_id) {
    mono_trace_result result;

    dist_debug_set_session(session_id);
    dist_debug_set_worker("monolithic", "local");
    dist_debug_reset_recorder("monolithic");
    trace_recorder * rec = dist_debug_recorder();
    if (rec == nullptr) {
        result.error = "debug mode not enabled";
        return result;
    }

    ggml_backend_load_all();
    llama_model * model = llama_model_load_from_file(model_path.c_str(), llama_model_default_params());
    if (!model) {
        result.error = "model load failed";
        return result;
    }

    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx    = 512;
    cparams.n_batch  = 512;
    cparams.no_perf  = true;
    cparams.embeddings = true;
    llama_context * ctx = llama_init_from_model(model, cparams);
    if (!ctx) {
        llama_model_free(model);
        result.error = "context init failed";
        return result;
    }
    llama_set_embeddings(ctx, true);

    const llama_vocab * vocab = llama_model_get_vocab(model);
    const int32_t n_vocab     = llama_vocab_n_tokens(vocab);
    const int32_t n_embd      = llama_model_n_embd(model);
    const auto prompt_tokens  = split_gen_tokenize(vocab, prompt);
    if (prompt_tokens.empty()) {
        llama_free(ctx);
        llama_model_free(model);
        result.error = "empty prompt";
        return result;
    }

    for (const auto t : prompt_tokens) {
        result.prompt_tokens.push_back(static_cast<int32_t>(t));
    }

    llama_sampler * smpl = split_gen_make_sampler();
    int32_t step = 0;

    rec->emit_step_begin(step, "prefill", -1, 0, 0);
    dist_debug_log_position(rec, step, "prefill", ctx, 0, (int32_t) prompt_tokens.size(), 1);

    if (split_gen_decode_tokens(ctx, prompt_tokens, 0, true) != 0) {
        llama_sampler_free(smpl);
        llama_free(ctx);
        llama_model_free(model);
        result.error = "prefill decode failed";
        return result;
    }

    dist_debug_log_kv(rec, step, "prefill", ctx, 0);
    dist_debug_log_hidden_out(rec, step, "prefill", ctx, (int32_t) prompt_tokens.size(), n_embd, "mono");
    dist_debug_log_logits_out(rec, step, "prefill", ctx, n_vocab, (int32_t) prompt_tokens.size() - 1);

    mono_decode_step prefill{};
    prefill.step     = step;
    prefill.phase    = "prefill";
    prefill.position = 0;
    const float * hidden = llama_get_embeddings(ctx);
    if (hidden) {
        const int64_t hn = static_cast<int64_t>(prompt_tokens.size()) * n_embd;
        prefill.hidden.assign(hidden, hidden + hn);
        prefill.hidden_stats = compute_tensor_stats(prefill.hidden.data(), hn);
    }
    const float * logits = llama_get_logits_ith(ctx, (int32_t) prompt_tokens.size() - 1);
    if (logits) {
        prefill.logits.assign(logits, logits + n_vocab);
        prefill.logits_stats = compute_tensor_stats(prefill.logits.data(), n_vocab);
        prefill.argmax_token = argmax_logits(logits, n_vocab);
        prefill.argmax_score = prefill.argmax_token >= 0 ? logits[prefill.argmax_token] : 0.0f;
    }

    bool used_argmax = false;
    const int32_t first_tok = dist_debug_sample_or_argmax(
            smpl, ctx, (int32_t) prompt_tokens.size() - 1, &used_argmax);
    if (!used_argmax) {
        llama_sampler_accept(smpl, first_tok);
    }
    prefill.selected_token = first_tok;
    rec->emit_token_selected(step, "prefill", first_tok, (int32_t) prompt_tokens.size() - 1, used_argmax);
    result.steps.push_back(std::move(prefill));
    step++;

    llama_token cur = first_tok;
    const int n_prompt = (int) prompt_tokens.size();
    for (int i = 1; i < max_tokens; ++i) {
        const int32_t pos = n_prompt + i - 1;
        rec->emit_step_begin(step, "decode", cur, pos, 0);
        dist_debug_log_position(rec, step, "decode", ctx, pos, 1, 1);

        if (split_gen_decode_one(ctx, cur, pos) != 0) {
            break;
        }

        dist_debug_log_kv(rec, step, "decode", ctx, 0);
        dist_debug_log_hidden_out(rec, step, "decode", ctx, 1, n_embd, "mono");
        dist_debug_log_logits_out(rec, step, "decode", ctx, n_vocab, -1);

        mono_decode_step ds{};
        ds.step        = step;
        ds.phase       = "decode";
        ds.input_token = cur;
        ds.position    = pos;
        if (const float * h = llama_get_embeddings(ctx)) {
            ds.hidden.assign(h, h + n_embd);
            ds.hidden_stats = compute_tensor_stats(ds.hidden.data(), n_embd);
        }
        if (const float * lg = llama_get_logits_ith(ctx, -1)) {
            ds.logits.assign(lg, lg + n_vocab);
            ds.logits_stats = compute_tensor_stats(ds.logits.data(), n_vocab);
            ds.argmax_token = argmax_logits(lg, n_vocab);
            ds.argmax_score = ds.argmax_token >= 0 ? lg[ds.argmax_token] : 0.0f;
        }
        llama_memory_t mem = llama_get_memory(ctx);
        const llama_pos seq_max = mem ? llama_memory_seq_pos_max(mem, 0) : -1;
        ds.kv_entries = seq_max >= 0 ? static_cast<int32_t>(seq_max + 1) : 0;
        ds.seq_len    = ds.kv_entries;

        used_argmax = false;
        cur = dist_debug_sample_or_argmax(smpl, ctx, -1, &used_argmax);
        if (!used_argmax) {
            llama_sampler_accept(smpl, cur);
        }
        ds.selected_token = cur;
        rec->emit_token_selected(step, "decode", cur, pos, used_argmax);
        result.steps.push_back(std::move(ds));
        step++;
    }

    llama_sampler_free(smpl);
    llama_free(ctx);
    llama_model_free(model);

    result.trace_path = rec->trace_path();
    result.ok         = true;
    return result;
}

mono_trace_result load_trace_file(const std::string & jsonl_path) {
    mono_trace_result result;
    result.trace_path = jsonl_path;
    std::ifstream in(jsonl_path);
    if (!in) {
        result.error = "cannot open trace: " + jsonl_path;
        return result;
    }
    result.ok = true;
    return result;
}
