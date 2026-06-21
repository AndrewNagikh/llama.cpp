// Controller: full-model reference vs split autoregressive generation (A -> TCP -> B)

#include "ggml-backend.h"
#include "ggml.h"
#include "llama.h"
#include "split_gen_common.h"
#include "split_tcp_wire.h"

#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#if !defined(_WIN32)
#include <signal.h>
#include <sys/wait.h>
#include <unistd.h>
#endif

static std::string exe_dir(const char * argv0) {
    std::string p(argv0);
    const auto pos = p.find_last_of("/\\");
    if (pos != std::string::npos) {
        return p.substr(0, pos + 1);
    }
    return "./";
}

struct gen_step {
    llama_token id;
    std::string text;
};

struct gen_result {
    std::vector<gen_step> steps;
    double ms_total = 0.0;
};

static gen_result run_full(
        llama_context * ctx,
        const llama_vocab * vocab,
        llama_sampler * smpl,
        const std::vector<llama_token> & prompt,
        int max_new) {
    gen_result out;
    const int64_t t0 = ggml_time_us();

    llama_set_layer_range(ctx, 0, -1);
    llama_clear_hidden_state(ctx);
    llama_memory_clear(llama_get_memory(ctx), true);

    const int n_prompt = (int) prompt.size();
    if (split_gen_decode_tokens(ctx, prompt, 0, false) != 0) {
        fprintf(stderr, "full: prefill failed\n");
        return out;
    }

    for (int step = 0; step < max_new; ++step) {
        const llama_token id = llama_sampler_sample(smpl, ctx, -1);
        llama_sampler_accept(smpl, id);

        gen_step gs;
        gs.id   = id;
        gs.text = split_gen_token_text(vocab, id);
        out.steps.push_back(gs);

        printf("FULL step=%d token=%d text=%s\n", step, (int) id, gs.text.c_str());
        fflush(stdout);

        if (llama_vocab_is_eog(vocab, id)) {
            break;
        }

        if (step + 1 >= max_new) {
            break;
        }

        const llama_pos pos = n_prompt + step;
        if (split_gen_decode_one(ctx, id, pos) != 0) {
            fprintf(stderr, "full: decode failed step=%d\n", step);
            break;
        }
    }

    out.ms_total = (ggml_time_us() - t0) / 1000.0;
    return out;
}

#if !defined(_WIN32)

static bool gen_send_recv(
        int ctrl_fd,
        split_gen_cmd cmd,
        int32_t n_tokens,
        int32_t pos_start,
        int32_t layer_end,
        bool include_logits,
        const int32_t * tokens,
        split_gen_a_resp & resp,
        std::vector<float> * logits) {
    if (!split_gen_send_req(ctrl_fd, cmd, n_tokens, pos_start, layer_end, include_logits ? 1 : 0, tokens)) {
        return false;
    }
    return split_gen_recv_resp(ctrl_fd, resp, logits);
}

static gen_result run_split(
        int ctrl_fd,
        const std::vector<llama_token> & prompt,
        int max_new,
        int layer_end,
        double & ms_a_sum,
        double & ms_b_sum,
        double & ms_xfer_sum) {
    gen_result out;
    const int64_t t0 = ggml_time_us();

    split_gen_a_resp resp{};

    if (!gen_send_recv(ctrl_fd, SPLIT_GEN_CMD_RESET, 0, 0, layer_end, false, nullptr, resp, nullptr)) {
        fprintf(stderr, "split: reset failed\n");
        return out;
    }

    std::vector<int32_t> ptoks(prompt.size());
    for (size_t i = 0; i < prompt.size(); ++i) {
        ptoks[i] = (int32_t) prompt[i];
    }

    if (!gen_send_recv(ctrl_fd, SPLIT_GEN_CMD_PREFILL, (int32_t) prompt.size(), 0, layer_end,
            false, ptoks.data(), resp, nullptr)) {
        fprintf(stderr, "split: prefill failed\n");
        return out;
    }

    ms_a_sum    += resp.ms_a_compute;
    ms_b_sum    += resp.ms_b_compute;
    ms_xfer_sum += resp.ms_ab_xfer;

    gen_step gs0;
    gs0.id = (llama_token) resp.token_id;
    out.steps.push_back(gs0);
    printf("SPLIT step=0 token=%d\n", resp.token_id);
    fflush(stdout);

    const int n_prompt = (int) prompt.size();
    llama_token cur = gs0.id;

    for (int step = 1; step < max_new; ++step) {
        const int32_t tok_i32 = (int32_t) cur;
        const int32_t pos     = n_prompt + step - 1;

        if (!gen_send_recv(ctrl_fd, SPLIT_GEN_CMD_DECODE, 1, pos, layer_end,
                false, &tok_i32, resp, nullptr)) {
            fprintf(stderr, "split: decode failed step=%d\n", step);
            break;
        }

        ms_a_sum    += resp.ms_a_compute;
        ms_b_sum    += resp.ms_b_compute;
        ms_xfer_sum += resp.ms_ab_xfer;

        gen_step gs;
        gs.id = (llama_token) resp.token_id;
        out.steps.push_back(gs);
        printf("SPLIT step=%d token=%d\n", step, resp.token_id);
        fflush(stdout);

        cur = gs.id;
    }

    gen_send_recv(ctrl_fd, SPLIT_GEN_CMD_SHUTDOWN, 0, 0, layer_end, false, nullptr, resp, nullptr);

    out.ms_total = (ggml_time_us() - t0) / 1000.0;
    return out;
}

static std::vector<llama_token> load_prompt_tokens(const char * model_path) {
    ggml_backend_load_all();
    llama_model * model = llama_model_load_from_file(model_path, llama_model_default_params());
    if (!model) {
        return {};
    }
    const auto toks = split_gen_tokenize(llama_model_get_vocab(model), SPLIT_GEN_PROMPT);
    llama_model_free(model);
    return toks;
}

#endif

int main(int argc, char ** argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: %s MODEL.gguf\n", argv[0]);
        return 1;
    }

#if defined(_WIN32)
    fprintf(stderr, "test-autoregressive-split requires fork() (Unix)\n");
    return 1;
#else
    const char * model_path = argv[1];
    const int ab_port   = 20000 + (getpid() % 1000);
    const int ctrl_port = ab_port + 1;
    const int layer_end = SPLIT_GEN_LAYER_END;
    const int max_new     = SPLIT_GEN_MAX_NEW;

    // tokenize prompt without keeping a llama context in parent (fork-safe)
    const auto prompt = load_prompt_tokens(model_path);
    if (prompt.empty()) {
        fprintf(stderr, "failed to tokenize prompt\n");
        return 1;
    }

    printf("prompt=\"%s\" n_prompt=%zu max_new=%d layer_end=%d\n",
            SPLIT_GEN_PROMPT, prompt.size(), max_new, layer_end);
    fflush(stdout);

    const std::string dir      = exe_dir(argv[0]);
    const std::string b_bin    = dir + "split_gen_b";
    const std::string a_bin    = dir + "split_gen_a";
    const std::string ab_str   = std::to_string(ab_port);
    const std::string ctrl_str = std::to_string(ctrl_port);
    const std::string le_str   = std::to_string(layer_end);

    const pid_t pid_b = fork();
    if (pid_b == 0) {
        execl(b_bin.c_str(), "split_gen_b", model_path,
                "--ab-port", ab_str.c_str(),
                "--layer-start", le_str.c_str(),
                (char *) nullptr);
        _exit(127);
    }

    usleep(300000);

    const pid_t pid_a = fork();
    if (pid_a == 0) {
        execl(a_bin.c_str(), "split_gen_a", model_path,
                "--ctrl-port", ctrl_str.c_str(),
                "--b-port", ab_str.c_str(),
                "--layer-end", le_str.c_str(),
                (char *) nullptr);
        _exit(127);
    }

    int ctrl_fd = -1;
    for (int retry = 0; retry < 60; ++retry) {
        ctrl_fd = split_tcp_connect("127.0.0.1", ctrl_port);
        if (ctrl_fd >= 0) {
            break;
        }
        usleep(100000);
    }
    if (ctrl_fd < 0) {
        fprintf(stderr, "controller: connect to A failed\n");
        kill(pid_a, SIGTERM);
        kill(pid_b, SIGTERM);
        return 1;
    }

    double ms_a = 0, ms_b = 0, ms_xfer = 0;
    gen_result split_res = run_split(ctrl_fd, prompt, max_new, layer_end, ms_a, ms_b, ms_xfer);

    close(ctrl_fd);

    int st = 0;
    waitpid(pid_a, &st, 0);
    waitpid(pid_b, &st, 0);

    // reload vocab for text conversion in full path
    ggml_backend_load_all();
    llama_model * model = llama_model_load_from_file(model_path, llama_model_default_params());
    if (!model) {
        fprintf(stderr, "failed to load model for full reference\n");
        return 1;
    }
    const llama_vocab * vocab = llama_model_get_vocab(model);

    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx   = (uint32_t) (prompt.size() + max_new + 8);
    cparams.n_batch = (uint32_t) std::max(prompt.size(), (size_t) 512);
    cparams.no_perf = true;
    llama_context * ctx = llama_init_from_model(model, cparams);
    llama_sampler * smpl = split_gen_make_sampler();

    gen_result full = run_full(ctx, vocab, smpl, prompt, max_new);

    for (auto & s : split_res.steps) {
        s.text = split_gen_token_text(vocab, s.id);
    }
    for (auto & s : full.steps) {
        if (s.text.empty()) {
            s.text = split_gen_token_text(vocab, s.id);
        }
    }

    for (size_t i = 0; i < split_res.steps.size(); ++i) {
        printf("SPLIT step=%zu token=%d text=%s\n", i, (int) split_res.steps[i].id, split_res.steps[i].text.c_str());
    }

    llama_sampler_free(smpl);
    llama_free(ctx);
    llama_model_free(model);

    const size_t n_cmp = std::min(full.steps.size(), split_res.steps.size());
    for (size_t i = 0; i < n_cmp; ++i) {
        if (full.steps[i].id != split_res.steps[i].id) {
            fprintf(stderr, "MISMATCH step=%zu full_token=%d split_token=%d\n",
                    i, (int) full.steps[i].id, (int) split_res.steps[i].id);
            fprintf(stderr, "  full_text=%s\n", full.steps[i].text.c_str());
            fprintf(stderr, "  split_text=%s\n", split_res.steps[i].text.c_str());
            return 1;
        }
    }

    if (full.steps.size() != split_res.steps.size()) {
        fprintf(stderr, "MISMATCH length full=%zu split=%zu\n", full.steps.size(), split_res.steps.size());
        return 1;
    }

    std::string full_text;
    std::string split_text;
    for (const auto & s : full.steps) {
        full_text += s.text;
    }
    for (const auto & s : split_res.steps) {
        split_text += s.text;
    }

    printf("\nFULL text: %s\n", full_text.c_str());
    printf("SPLIT text: %s\n", split_text.c_str());
    printf("MATCH = %s\n", full_text == split_text ? "TRUE" : "FALSE");

    const double full_tps  = full.steps.size()  / (full.ms_total  / 1000.0);
    const double split_tps = split_res.steps.size() / (split_res.ms_total / 1000.0);

    printf("perf: full_ms=%.3f split_ms=%.3f\n", full.ms_total, split_res.ms_total);
    printf("perf: full_tps=%.3f split_tps=%.3f\n", full_tps, split_tps);
    printf("perf: split_a_ms=%.3f split_b_ms=%.3f split_tcp_ms=%.3f\n", ms_a, ms_b, ms_xfer);

    printf("test-autoregressive-split: OK (%zu tokens)\n", full.steps.size());
    return 0;
#endif
}
