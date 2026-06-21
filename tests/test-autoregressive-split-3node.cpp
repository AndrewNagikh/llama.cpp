// Controller: 3-node split generation across layouts A/B/C vs full model

#include "ggml-backend.h"
#include "ggml.h"
#include "llama.h"
#include "split_gen3_common.h"
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
    std::vector<split_gen3_step_perf> perf;
    double ms_total = 0.0;
};

struct layout_result {
    const char * name;
    gen_result split;
    double tps = 0.0;
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

static bool gen3_send_recv(
        int ctrl_fd,
        split_gen_cmd cmd,
        int32_t n_tokens,
        int32_t pos_start,
        int32_t layer_end,
        bool include_logits,
        const int32_t * tokens,
        split_gen3_a_resp & resp,
        std::vector<float> * logits) {
    if (!split_gen_send_req(ctrl_fd, cmd, n_tokens, pos_start, layer_end, include_logits ? 1 : 0, tokens)) {
        return false;
    }
    return split_gen3_recv_a_resp(ctrl_fd, resp, logits);
}

static void print_step_perf(int step, const split_gen3_a_resp & resp) {
    const double total = resp.ms_a_compute + resp.ms_ab_xfer + resp.ms_b_compute +
                           resp.ms_bc_xfer + resp.ms_c_compute + resp.ms_c_sample;
    printf("  perf step=%d A=%.3f AB=%.3f B=%.3f BC=%.3f C=%.3f sample=%.3f total=%.3f ms\n",
            step,
            resp.ms_a_compute, resp.ms_ab_xfer, resp.ms_b_compute,
            resp.ms_bc_xfer, resp.ms_c_compute, resp.ms_c_sample, total);
    fflush(stdout);
}

static gen_result run_split3(
        int ctrl_fd,
        const std::vector<llama_token> & prompt,
        int max_new,
        int layer_a_end) {
    gen_result out;
    const int64_t t0 = ggml_time_us();

    split_gen3_a_resp resp{};

    if (!gen3_send_recv(ctrl_fd, SPLIT_GEN_CMD_RESET, 0, 0, layer_a_end, false, nullptr, resp, nullptr)) {
        fprintf(stderr, "split3: reset failed\n");
        return out;
    }

    std::vector<int32_t> ptoks(prompt.size());
    for (size_t i = 0; i < prompt.size(); ++i) {
        ptoks[i] = (int32_t) prompt[i];
    }

    if (!gen3_send_recv(ctrl_fd, SPLIT_GEN_CMD_PREFILL, (int32_t) prompt.size(), 0, layer_a_end,
            false, ptoks.data(), resp, nullptr)) {
        fprintf(stderr, "split3: prefill failed\n");
        return out;
    }

    split_gen3_step_perf p0;
    p0.ms_a      = resp.ms_a_compute;
    p0.ms_ab     = resp.ms_ab_xfer;
    p0.ms_b      = resp.ms_b_compute;
    p0.ms_bc     = resp.ms_bc_xfer;
    p0.ms_c      = resp.ms_c_compute;
    p0.ms_sample = resp.ms_c_sample;
    p0.ms_total  = p0.ms_a + p0.ms_ab + p0.ms_b + p0.ms_bc + p0.ms_c + p0.ms_sample;
    out.perf.push_back(p0);
    print_step_perf(0, resp);

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

        if (!gen3_send_recv(ctrl_fd, SPLIT_GEN_CMD_DECODE, 1, pos, layer_a_end,
                false, &tok_i32, resp, nullptr)) {
            fprintf(stderr, "split3: decode failed step=%d\n", step);
            break;
        }

        split_gen3_step_perf ps;
        ps.ms_a      = resp.ms_a_compute;
        ps.ms_ab     = resp.ms_ab_xfer;
        ps.ms_b      = resp.ms_b_compute;
        ps.ms_bc     = resp.ms_bc_xfer;
        ps.ms_c      = resp.ms_c_compute;
        ps.ms_sample = resp.ms_c_sample;
        ps.ms_total  = ps.ms_a + ps.ms_ab + ps.ms_b + ps.ms_bc + ps.ms_c + ps.ms_sample;
        out.perf.push_back(ps);
        print_step_perf(step, resp);

        gen_step gs;
        gs.id = (llama_token) resp.token_id;
        out.steps.push_back(gs);
        printf("SPLIT step=%d token=%d\n", step, resp.token_id);
        fflush(stdout);

        cur = gs.id;
    }

    gen3_send_recv(ctrl_fd, SPLIT_GEN_CMD_SHUTDOWN, 0, 0, layer_a_end, false, nullptr, resp, nullptr);

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

static int fork_pipeline(
        const char * model_path,
        const std::string & dir,
        const split_gen3_layout & layout,
        int bc_port,
        int ab_port,
        int ctrl_port,
        pid_t & pid_a,
        pid_t & pid_b,
        pid_t & pid_c) {
    const std::string c_bin = dir + "split_gen3_c";
    const std::string b_bin = dir + "split_gen3_b";
    const std::string a_bin = dir + "split_gen3_a";

    const std::string bc_str   = std::to_string(bc_port);
    const std::string ab_str   = std::to_string(ab_port);
    const std::string ctrl_str = std::to_string(ctrl_port);
    const std::string ls_c     = std::to_string(layout.layer_c_start);
    const std::string ls_b     = std::to_string(layout.layer_b_start);
    const std::string le_b     = std::to_string(layout.layer_b_end);
    const std::string le_a     = std::to_string(layout.layer_a_end);

    pid_c = fork();
    if (pid_c == 0) {
        execl(c_bin.c_str(), "split_gen3_c", model_path,
                "--bc-port", bc_str.c_str(),
                "--layer-start", ls_c.c_str(),
                (char *) nullptr);
        _exit(127);
    }

    usleep(500000);

    pid_b = fork();
    if (pid_b == 0) {
        execl(b_bin.c_str(), "split_gen3_b", model_path,
                "--ab-port", ab_str.c_str(),
                "--bc-port", bc_str.c_str(),
                "--layer-start", ls_b.c_str(),
                "--layer-end", le_b.c_str(),
                (char *) nullptr);
        _exit(127);
    }

    usleep(500000);

    pid_a = fork();
    if (pid_a == 0) {
        execl(a_bin.c_str(), "split_gen3_a", model_path,
                "--ctrl-port", ctrl_str.c_str(),
                "--b-port", ab_str.c_str(),
                "--layer-end", le_a.c_str(),
                (char *) nullptr);
        _exit(127);
    }

    return 0;
}

static bool compare_tokens(
        const gen_result & full,
        gen_result & split,
        const llama_vocab * vocab) {
    for (auto & s : split.steps) {
        s.text = split_gen_token_text(vocab, s.id);
    }

    for (size_t i = 0; i < split.steps.size(); ++i) {
        printf("SPLIT step=%zu token=%d text=%s\n", i, (int) split.steps[i].id, split.steps[i].text.c_str());
    }

    const size_t n_cmp = std::min(full.steps.size(), split.steps.size());
    for (size_t i = 0; i < n_cmp; ++i) {
        if (full.steps[i].id != split.steps[i].id) {
            fprintf(stderr, "MISMATCH step=%zu full_token=%d split_token=%d\n",
                    i, (int) full.steps[i].id, (int) split.steps[i].id);
            fprintf(stderr, "  full_text=%s\n", full.steps[i].text.c_str());
            fprintf(stderr, "  split_text=%s\n", split.steps[i].text.c_str());
            return false;
        }
    }

    if (full.steps.size() != split.steps.size()) {
        fprintf(stderr, "MISMATCH length full=%zu split=%zu\n", full.steps.size(), split.steps.size());
        return false;
    }

    std::string full_text;
    std::string split_text;
    for (const auto & s : full.steps) {
        full_text += s.text;
    }
    for (const auto & s : split.steps) {
        split_text += s.text;
    }

    printf("\nFULL text: %s\n", full_text.c_str());
    printf("SPLIT text: %s\n", split_text.c_str());
    printf("MATCH = %s\n", full_text == split_text ? "TRUE" : "FALSE");
    return full_text == split_text;
}

#endif

int main(int argc, char ** argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: %s MODEL.gguf\n", argv[0]);
        return 1;
    }

#if defined(_WIN32)
    fprintf(stderr, "test-autoregressive-split-3node requires fork() (Unix)\n");
    return 1;
#else
    const char * model_path = argv[1];
    const int max_new = SPLIT_GEN3_MAX_NEW;
    const int base_port = 21000 + (getpid() % 500);

    const auto prompt = load_prompt_tokens(model_path);
    if (prompt.empty()) {
        fprintf(stderr, "failed to tokenize prompt\n");
        return 1;
    }

    printf("prompt=\"%s\" n_prompt=%zu max_new=%d\n", SPLIT_GEN_PROMPT, prompt.size(), max_new);
    fflush(stdout);

    const std::string dir = exe_dir(argv[0]);
    layout_result results[SPLIT_GEN3_N_LAYOUTS];

    for (int li = 0; li < SPLIT_GEN3_N_LAYOUTS; ++li) {
        const split_gen3_layout & layout = SPLIT_GEN3_LAYOUTS[li];
        results[li].name = layout.name;

        const int bc_port   = base_port + li * 10;
        const int ab_port   = bc_port + 1;
        const int ctrl_port = bc_port + 2;

        printf("\n=== layout %s: A=[0,%d) B=[%d,%d) C=[%d,16) ===\n",
                layout.name, layout.layer_a_end,
                layout.layer_b_start, layout.layer_b_end,
                layout.layer_c_start);
        fflush(stdout);

        pid_t pid_a = 0, pid_b = 0, pid_c = 0;
        fork_pipeline(model_path, dir, layout, bc_port, ab_port, ctrl_port, pid_a, pid_b, pid_c);

        int ctrl_fd = -1;
        for (int retry = 0; retry < 300; ++retry) {
            ctrl_fd = split_tcp_connect("127.0.0.1", ctrl_port);
            if (ctrl_fd >= 0) {
                break;
            }
            usleep(100000);
        }
        if (ctrl_fd < 0) {
            fprintf(stderr, "controller: connect to A failed layout=%s\n", layout.name);
            kill(pid_a, SIGTERM);
            kill(pid_b, SIGTERM);
            kill(pid_c, SIGTERM);
            return 1;
        }

        results[li].split = run_split3(ctrl_fd, prompt, max_new, layout.layer_a_end);
        close(ctrl_fd);

        int st = 0;
        waitpid(pid_a, &st, 0);
        waitpid(pid_b, &st, 0);
        waitpid(pid_c, &st, 0);

        results[li].tps = results[li].split.perf.empty() ? 0.0 :
            results[li].split.steps.size() / (results[li].split.ms_total / 1000.0);

        double sum_a = 0, sum_ab = 0, sum_b = 0, sum_bc = 0, sum_c = 0, sum_s = 0;
        for (const auto & p : results[li].split.perf) {
            sum_a  += p.ms_a;
            sum_ab += p.ms_ab;
            sum_b  += p.ms_b;
            sum_bc += p.ms_bc;
            sum_c  += p.ms_c;
            sum_s  += p.ms_sample;
        }
        printf("layout %s summary: tokens=%zu wall_ms=%.3f tps=%.3f\n",
                layout.name, results[li].split.steps.size(), results[li].split.ms_total, results[li].tps);
        printf("  avg A=%.3f AB=%.3f B=%.3f BC=%.3f C=%.3f sample=%.3f ms/token\n",
                results[li].split.perf.empty() ? 0.0 : sum_a  / results[li].split.perf.size(),
                results[li].split.perf.empty() ? 0.0 : sum_ab / results[li].split.perf.size(),
                results[li].split.perf.empty() ? 0.0 : sum_b  / results[li].split.perf.size(),
                results[li].split.perf.empty() ? 0.0 : sum_bc / results[li].split.perf.size(),
                results[li].split.perf.empty() ? 0.0 : sum_c  / results[li].split.perf.size(),
                results[li].split.perf.empty() ? 0.0 : sum_s  / results[li].split.perf.size());
    }

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

    printf("\n=== full model reference ===\n");
    gen_result full = run_full(ctx, vocab, smpl, prompt, max_new);
    const double full_tps = full.steps.size() / (full.ms_total / 1000.0);

    bool all_ok = true;
    for (int li = 0; li < SPLIT_GEN3_N_LAYOUTS; ++li) {
        printf("\n=== layout %s correctness ===\n", results[li].name);
        if (!compare_tokens(full, results[li].split, vocab)) {
            all_ok = false;
        }
    }

    llama_sampler_free(smpl);
    llama_free(ctx);
    llama_model_free(model);

    printf("\n=== final report ===\n");
    printf("Layout      TPS\n");
    for (int li = 0; li < SPLIT_GEN3_N_LAYOUTS; ++li) {
        printf("%-11s %.3f\n", results[li].name, results[li].tps);
    }
    printf("FULL        %.3f\n", full_tps);

    int best_idx = 0;
    for (int li = 1; li < SPLIT_GEN3_N_LAYOUTS; ++li) {
        if (results[li].tps > results[best_idx].tps) {
            best_idx = li;
        }
    }
    printf("Best layout = %s (%.3f tps)\n", results[best_idx].name, results[best_idx].tps);

    if (!all_ok) {
        fprintf(stderr, "test-autoregressive-split-3node: FAILED\n");
        return 1;
    }

    printf("test-autoregressive-split-3node: OK (%zu tokens x %d layouts)\n",
            full.steps.size(), SPLIT_GEN3_N_LAYOUTS);
    return 0;
#endif
}
