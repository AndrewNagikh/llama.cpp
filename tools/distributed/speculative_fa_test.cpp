// Task 19 Phase 3 driver: validates the fa-link placement (draft colocated
// with `final`, per TASK_19_SPECULATIVE_PIPELINE_STUDY.md SA) end to end,
// exactly as node_agent's speculative client loop drives it: the client
// only ever sends a 1-token anchor via SPLIT_GEN_CMD_VERIFY; entry silently
// extends the wave from whatever final has shipped over the fa-link.
//
// Usage:
//   speculative_fa_test TARGET.gguf DRAFT.gguf [N_PREDICT] [K] [A_END] [B_END]

#include "ggml.h"
#include "llama.h"
#include "transport/split_tcp_wire.h"
#include "workers/split_gen_common.h"

#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#if defined(_WIN32)
int main() {
    fprintf(stderr, "speculative_fa_test: POSIX only\n");
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

int main(int argc, char ** argv) {
    if (argc < 3) {
        fprintf(stderr, "usage: %s TARGET.gguf DRAFT.gguf [N_PREDICT] [K] [A_END] [B_END]\n", argv[0]);
        return 1;
    }
    const char * target_path = argv[1];
    const char * draft_path  = argv[2];
    const int n_predict = argc > 3 ? atoi(argv[3]) : 48;
    const int k         = argc > 4 ? atoi(argv[4]) : 4;
    const int a_end     = argc > 5 ? atoi(argv[5]) : 9;
    const int b_end     = argc > 6 ? atoi(argv[6]) : 18;
    const std::string prompt = "Once upon a time, in a small village near the mountains, there lived";

    setenv("DIST_RUNTIME_STAGE_QUEUE", "1", 1);
    setenv("DIST_RUNTIME_ENTRY_QUEUE", "0", 1);

    std::string dir = argv[0];
    const size_t slash = dir.find_last_of('/');
    dir = slash == std::string::npos ? "./" : dir.substr(0, slash + 1);

    const int base      = 27000 + (getpid() % 500);
    const int bc_port   = base;
    const int ab_port   = base + 1;
    const int ctrl_port = base + 2;
    const int fa_port   = base + 3;

    pid_t pid_c = fork();
    if (pid_c == 0) {
        const std::string bc = std::to_string(bc_port);
        const std::string ls = std::to_string(b_end);
        const std::string fa = std::to_string(fa_port);
        const std::string dk = std::to_string(k);
        execl((dir + "split_gen3_c").c_str(), "split_gen3_c", target_path,
                "--bc-port", bc.c_str(), "--layer-start", ls.c_str(),
                "--draft-model", draft_path, "--fa-host", "127.0.0.1", "--fa-port", fa.c_str(),
                "--draft-k", dk.c_str(), (char *) nullptr);
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
        const std::string fa   = std::to_string(fa_port);
        execl((dir + "split_gen3_a").c_str(), "split_gen3_a", target_path,
                "--ctrl-port", ctrl.c_str(), "--b-port", ab.c_str(),
                "--layer-end", le.c_str(), "--fa-port", fa.c_str(), (char *) nullptr);
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

    // Give final time to load its draft model and connect the fa-link back
    // to entry before we prefill (best-effort in the real system; generous
    // here so the test actually exercises drafting instead of the fallback).
    sleep(2);

    ggml_backend_load_all();
    llama_model * model = llama_model_load_from_file(target_path, llama_model_default_params());
    if (!model) {
        fprintf(stderr, "target model load failed\n");
        cleanup();
        return 1;
    }
    const auto ptokens = split_gen_tokenize(llama_model_get_vocab(model), prompt);
    llama_model_free(model);
    std::vector<int32_t> ptoks(ptokens.size());
    for (size_t i = 0; i < ptokens.size(); ++i) {
        ptoks[i] = (int32_t) ptokens[i];
    }
    const int32_t n_prompt = (int32_t) ptoks.size();

    split_gen3_a_resp resp{};

    // ---- baseline: plain greedy decode (no speculation used) ----
    send_recv(ctrl_fd, SPLIT_GEN_CMD_RESET, 0, 0, a_end, nullptr, resp);
    if (!send_recv(ctrl_fd, SPLIT_GEN_CMD_PREFILL, n_prompt, 0, a_end, ptoks.data(), resp)) {
        fprintf(stderr, "baseline prefill failed\n");
        cleanup();
        return 1;
    }
    std::vector<int32_t> base_out;
    base_out.push_back(resp.token_id);
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

    // ---- speculative: client only ever sends a 1-token anchor via VERIFY;
    // entry silently extends the wave from final's fa-link draft buffer. ----
    send_recv(ctrl_fd, SPLIT_GEN_CMD_RESET, 0, 0, a_end, nullptr, resp);
    if (!send_recv(ctrl_fd, SPLIT_GEN_CMD_PREFILL, n_prompt, 0, a_end, ptoks.data(), resp)) {
        fprintf(stderr, "spec prefill failed\n");
        cleanup();
        return 1;
    }

    std::vector<int32_t> spec_out;
    spec_out.push_back(resp.token_id);
    int64_t n_waves = 0, n_accepted = 0;
    const int64_t t0 = ggml_time_us();
    {
        int32_t cur = resp.token_id;
        while ((int) spec_out.size() < n_predict) {
            const int32_t pos = n_prompt + (int32_t) spec_out.size() - 1;
            // Localhost has ~0 network RTT, so final's draft compute never
            // gets a head start the way it would behind a real return-path
            // R (see SA). Simulate R here, client-side only, just to prove
            // the accept path itself (not only the fallback) is correct.
            usleep(80000);
            if (!send_recv(ctrl_fd, SPLIT_GEN_CMD_VERIFY, 1, pos, a_end, &cur, resp)) {
                fprintf(stderr, "verify failed at token %zu\n", spec_out.size());
                cleanup();
                return 1;
            }
            n_waves++;
            const int32_t accepted = resp.accepted_count;
            if (accepted < 0 || accepted > SPLIT_GEN_SPEC_MAX_K) {
                fprintf(stderr, "bad accepted_count %d\n", accepted);
                cleanup();
                return 1;
            }
            n_accepted += accepted;
            for (int32_t i = 0; i < accepted && (int) spec_out.size() < n_predict; ++i) {
                spec_out.push_back(resp.accepted_ids[i]);
            }
            cur = resp.token_id;
            if ((int) spec_out.size() < n_predict) {
                spec_out.push_back(cur);
            }
        }
    }
    const double spec_s = (ggml_time_us() - t0) / 1e6;

    send_recv(ctrl_fd, SPLIT_GEN_CMD_SHUTDOWN, 0, 0, a_end, nullptr, resp);
    cleanup();

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

    printf("tokens        : %d\n", n_predict);
    printf("k             : %d\n", k);
    printf("waves         : %lld\n", (long long) n_waves);
    printf("speculative   : %.2f s (%.2f tok/s)\n", spec_s, (n_predict - 1) / spec_s);
    printf("draft accepted: %lld tokens across %lld waves (avg %.2f/wave)\n",
            (long long) n_accepted, (long long) n_waves,
            n_waves > 0 ? (double) n_accepted / (double) n_waves : 0.0);
    printf("determinism   : %s\n", match ? "PASS (streams identical)" : "FAIL");
    if (!match) {
        fprintf(stderr, "first diff at %zu: base=%d spec=%d\n",
                first_diff,
                first_diff < base_out.size() ? base_out[first_diff] : -1,
                first_diff < spec_out.size() ? spec_out[first_diff] : -1);
        return 1;
    }
    if (n_waves > 0 && n_accepted == 0) {
        fprintf(stderr, "warning: draft never accepted -- fa-link likely never connected in time\n");
        return 2;
    }
    return 0;
}

#endif
