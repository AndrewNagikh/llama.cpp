// Task 19 Phase 1 driver: runs the 3-stage pipeline locally and validates
// speculative verify waves against the plain greedy decode loop.
//
// Usage:
//   speculative_pipeline_test TARGET.gguf DRAFT.gguf [N_PREDICT] [K] [A_END] [B_END]
//
// The speculative token stream must match the baseline stream exactly
// (greedy verification reproduces the target model's own argmax chain).

#include "ggml.h"
#include "llama.h"
#include "workers/split_gen_common.h"
#include "transport/split_tcp_wire.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#if defined(_WIN32)
int main() {
    fprintf(stderr, "speculative_pipeline_test: POSIX only\n");
    return 77;
}
#else

#include <signal.h>
#include <sys/wait.h>
#include <unistd.h>

static void kill_wait(pid_t pid) {
    if (pid > 0) {
        kill(pid, SIGTERM);
        int status = 0;
        waitpid(pid, &status, 0);
    }
}

static bool send_recv(
        int ctrl_fd,
        split_gen_cmd cmd,
        int32_t n_tokens,
        int32_t pos_start,
        int32_t layer_end,
        const int32_t * tokens,
        split_gen3_a_resp & resp) {
    if (!split_gen_send_req(ctrl_fd, cmd, n_tokens, pos_start, layer_end, 0, tokens)) {
        return false;
    }
    return split_gen3_recv_a_resp(ctrl_fd, resp, nullptr);
}

static int32_t draft_argmax(llama_context * ctx, const int32_t n_vocab) {
    const float * logits = llama_get_logits_ith(ctx, -1);
    int32_t best = 0;
    for (int32_t v = 1; v < n_vocab; ++v) {
        if (logits[v] > logits[best]) {
            best = v;
        }
    }
    return best;
}

int main(int argc, char ** argv) {
    if (argc < 3) {
        fprintf(stderr, "usage: %s TARGET.gguf DRAFT.gguf [N_PREDICT] [K] [A_END] [B_END]\n", argv[0]);
        return 1;
    }
    const char * target_path = argv[1];
    const char * draft_path  = argv[2];
    const int n_predict = argc > 3 ? atoi(argv[3]) : 64;
    const int k         = argc > 4 ? atoi(argv[4]) : 4;
    const int a_end     = argc > 5 ? atoi(argv[5]) : 9;
    const int b_end     = argc > 6 ? atoi(argv[6]) : 18;
    const std::string prompt = "Once upon a time, in a small village near the mountains, there lived";

    // Verify waves need the synchronous entry path; stage queues stay on.
    setenv("DIST_RUNTIME_ENTRY_QUEUE", "0", 1);
    setenv("DIST_RUNTIME_CLIENT_PIPELINE", "0", 1);

    std::string dir = argv[0];
    const size_t slash = dir.find_last_of('/');
    dir = slash == std::string::npos ? "./" : dir.substr(0, slash + 1);

    const int base      = 26000 + (getpid() % 500);
    const int bc_port   = base;
    const int ab_port   = base + 1;
    const int ctrl_port = base + 2;

    pid_t pid_c = fork();
    if (pid_c == 0) {
        const std::string bc = std::to_string(bc_port);
        const std::string ls = std::to_string(b_end);
        execl((dir + "split_gen3_c").c_str(), "split_gen3_c", target_path,
                "--bc-port", bc.c_str(), "--layer-start", ls.c_str(), (char *) nullptr);
        _exit(127);
    }
    usleep(500000);

    pid_t pid_b = fork();
    if (pid_b == 0) {
        const std::string ab = std::to_string(ab_port);
        const std::string bc = std::to_string(bc_port);
        const std::string ls = std::to_string(a_end);
        const std::string le = std::to_string(b_end);
        execl((dir + "split_gen3_b").c_str(), "split_gen3_b", target_path,
                "--ab-port", ab.c_str(), "--bc-port", bc.c_str(),
                "--layer-start", ls.c_str(), "--layer-end", le.c_str(), (char *) nullptr);
        _exit(127);
    }
    usleep(500000);

    pid_t pid_a = fork();
    if (pid_a == 0) {
        const std::string ctrl = std::to_string(ctrl_port);
        const std::string ab   = std::to_string(ab_port);
        const std::string le   = std::to_string(a_end);
        execl((dir + "split_gen3_a").c_str(), "split_gen3_a", target_path,
                "--ctrl-port", ctrl.c_str(), "--b-port", ab.c_str(),
                "--layer-end", le.c_str(), (char *) nullptr);
        _exit(127);
    }

    int ctrl_fd = -1;
    for (int retry = 0; retry < 600 && ctrl_fd < 0; ++retry) {
        ctrl_fd = split_tcp_connect("127.0.0.1", ctrl_port);
        if (ctrl_fd < 0) {
            usleep(100000);
        }
    }
    auto cleanup = [&]() {
        if (ctrl_fd >= 0) {
            close(ctrl_fd);
        }
        kill_wait(pid_a);
        kill_wait(pid_b);
        kill_wait(pid_c);
    };
    if (ctrl_fd < 0) {
        fprintf(stderr, "ctrl connect failed\n");
        cleanup();
        return 1;
    }

    ggml_backend_load_all();

    llama_model_params dparams = llama_model_default_params();
    llama_model * draft_model = llama_model_load_from_file(draft_path, dparams);
    if (!draft_model) {
        fprintf(stderr, "draft model load failed\n");
        cleanup();
        return 1;
    }
    llama_context_params dcparams = llama_context_default_params();
    dcparams.n_ctx   = 1024;
    dcparams.n_batch = 512;
    llama_context * dctx = llama_init_from_model(draft_model, dcparams);
    if (!dctx) {
        fprintf(stderr, "draft context failed\n");
        cleanup();
        return 1;
    }
    const int32_t draft_n_vocab =
            llama_vocab_n_tokens(llama_model_get_vocab(draft_model));

    const auto ptokens = split_gen_tokenize(llama_model_get_vocab(draft_model), prompt);
    std::vector<int32_t> ptoks(ptokens.size());
    for (size_t i = 0; i < ptokens.size(); ++i) {
        ptoks[i] = (int32_t) ptokens[i];
    }
    const int32_t n_prompt = (int32_t) ptoks.size();

    split_gen3_a_resp resp{};

    // ---- baseline: plain greedy decode ----
    send_recv(ctrl_fd, SPLIT_GEN_CMD_RESET, 0, 0, a_end, nullptr, resp);
    if (!send_recv(ctrl_fd, SPLIT_GEN_CMD_PREFILL, n_prompt, 0, a_end, ptoks.data(), resp)) {
        fprintf(stderr, "baseline prefill failed\n");
        cleanup();
        return 1;
    }
    std::vector<int32_t> base_out;
    base_out.push_back(resp.token_id);
    const int64_t tb0 = ggml_time_us();
    {
        int32_t cur = resp.token_id;
        for (int step = 1; step < n_predict; ++step) {
            const int32_t pos = n_prompt + step - 1;
            if (!send_recv(ctrl_fd, SPLIT_GEN_CMD_DECODE, 1, pos, a_end, &cur, resp)) {
                fprintf(stderr, "baseline decode failed step=%d\n", step);
                cleanup();
                return 1;
            }
            cur = resp.token_id;
            base_out.push_back(cur);
        }
    }
    const double base_s = (ggml_time_us() - tb0) / 1e6;

    // ---- speculative: draft k, verify wave, repeat ----
    send_recv(ctrl_fd, SPLIT_GEN_CMD_RESET, 0, 0, a_end, nullptr, resp);
    if (!send_recv(ctrl_fd, SPLIT_GEN_CMD_PREFILL, n_prompt, 0, a_end, ptoks.data(), resp)) {
        fprintf(stderr, "spec prefill failed\n");
        cleanup();
        return 1;
    }

    // Draft prefill over the same prompt.
    {
        llama_batch batch = llama_batch_init(n_prompt, 0, 1);
        for (int32_t i = 0; i < n_prompt; ++i) {
            batch.token[i]     = (llama_token) ptoks[i];
            batch.pos[i]       = i;
            batch.n_seq_id[i]  = 1;
            batch.seq_id[i][0] = 0;
            batch.logits[i]    = (i == n_prompt - 1);
        }
        batch.n_tokens = n_prompt;
        if (llama_decode(dctx, batch) != 0) {
            fprintf(stderr, "draft prefill failed\n");
            llama_batch_free(batch);
            cleanup();
            return 1;
        }
        llama_batch_free(batch);
    }

    std::vector<int32_t> spec_out;
    spec_out.push_back(resp.token_id);
    int64_t n_drafted = 0, n_accepted = 0, n_waves = 0;
    int32_t draft_next_pos = n_prompt;

    const int64_t ts0 = ggml_time_us();
    {
        int32_t cur = resp.token_id;   // anchor: last accepted token, not yet decoded
        int32_t pos = n_prompt;        // its position
        while ((int) spec_out.size() < n_predict) {
            // Draft k tokens starting from the anchor.
            std::vector<int32_t> wave;
            wave.push_back(cur);
            // On full acceptance the draft never ingested the last accepted
            // draft token as an input; feed any such gap before continuing.
            while (draft_next_pos < pos) {
                const int32_t missed = spec_out[(size_t) (draft_next_pos - n_prompt)];
                if (split_gen_decode_one(dctx, (llama_token) missed, draft_next_pos) != 0) {
                    fprintf(stderr, "draft catch-up decode failed\n");
                    cleanup();
                    return 1;
                }
                draft_next_pos++;
            }
            if (pos < draft_next_pos) {
                llama_memory_seq_rm(llama_get_memory(dctx), 0, pos, -1);
            }
            int32_t dpos = pos;
            for (int d = 0; d < k; ++d) {
                if (split_gen_decode_one(dctx, (llama_token) wave.back(), dpos) != 0) {
                    fprintf(stderr, "draft decode failed\n");
                    cleanup();
                    return 1;
                }
                dpos++;
                wave.push_back(draft_argmax(dctx, draft_n_vocab));
            }
            draft_next_pos = dpos;
            n_drafted += k;

            if (!send_recv(ctrl_fd, SPLIT_GEN_CMD_VERIFY,
                        (int32_t) wave.size(), pos, a_end, wave.data(), resp)) {
                fprintf(stderr, "verify wave failed\n");
                cleanup();
                return 1;
            }
            n_waves++;
            const int32_t accepted = resp.accepted_count;
            if (accepted < 0 || accepted > k) {
                fprintf(stderr, "bad accepted_count %d\n", accepted);
                cleanup();
                return 1;
            }
            n_accepted += accepted;
            for (int32_t i = 0; i < accepted && (int) spec_out.size() < n_predict; ++i) {
                spec_out.push_back(wave[i + 1]);
            }
            if ((int) spec_out.size() < n_predict) {
                spec_out.push_back(resp.token_id);
            }
            pos = pos + accepted + 1;
            cur = resp.token_id;
        }
    }
    const double spec_s = (ggml_time_us() - ts0) / 1e6;

    send_recv(ctrl_fd, SPLIT_GEN_CMD_SHUTDOWN, 0, 0, a_end, nullptr, resp);
    llama_free(dctx);
    llama_model_free(draft_model);
    cleanup();

    // ---- report ----
    bool match = base_out.size() == spec_out.size();
    size_t first_diff = 0;
    if (match) {
        for (size_t i = 0; i < base_out.size(); ++i) {
            if (base_out[i] != spec_out[i]) {
                match = false;
                first_diff = i;
                break;
            }
        }
    }

    const double tps_base = (n_predict - 1) / base_s;
    const double tps_spec = (n_predict - 1) / spec_s;
    printf("tokens        : %d\n", n_predict);
    printf("k             : %d\n", k);
    printf("baseline      : %.2f s  (%.2f tok/s)\n", base_s, tps_base);
    printf("speculative   : %.2f s  (%.2f tok/s)\n", spec_s, tps_spec);
    printf("speedup       : x%.2f\n", tps_spec / tps_base);
    printf("acceptance    : %lld/%lld (%.1f%%), %lld waves\n",
            (long long) n_accepted, (long long) n_drafted,
            n_drafted > 0 ? 100.0 * (double) n_accepted / (double) n_drafted : 0.0,
            (long long) n_waves);
    printf("determinism   : %s\n", match ? "PASS (streams identical)" : "FAIL");
    if (!match) {
        fprintf(stderr, "first diff at %zu: base=%d spec=%d\n",
                first_diff,
                first_diff < base_out.size() ? base_out[first_diff] : -1,
                first_diff < spec_out.size() ? spec_out[first_diff] : -1);
        return 1;
    }
    return 0;
}

#endif
