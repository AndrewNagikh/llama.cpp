// Integration test: full model vs Process A (TCP) -> Process B

#include "ggml-backend.h"
#include "ggml.h"
#include "llama.h"
#include "split_tcp_wire.h"

#include "../src/llama-ext.h"

#include <cmath>
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

static int decode_tokens(llama_context * ctx, const std::vector<llama_token> & tokens) {
    const int32_t n_tokens = (int32_t) tokens.size();
    llama_batch batch = llama_batch_init(n_tokens, 0, 1);
    for (int32_t i = 0; i < n_tokens; ++i) {
        batch.token[i]     = tokens[i];
        batch.pos[i]       = i;
        batch.n_seq_id[i]  = 1;
        batch.seq_id[i][0] = 0;
        batch.logits[i]    = (i == n_tokens - 1);
    }
    batch.n_tokens = n_tokens;
    const int ret = llama_decode(ctx, batch);
    llama_batch_free(batch);
    return ret;
}

static std::vector<float> get_last_logits(llama_context * ctx, int n_vocab) {
    const float * logits = llama_get_logits_ith(ctx, -1);
    return std::vector<float>(logits, logits + n_vocab);
}

struct diff_stats {
    float max_abs = 0.0f;
    float mean_abs = 0.0f;
};

static diff_stats compare_vectors(const std::vector<float> & a, const std::vector<float> & b) {
    diff_stats s;
    if (a.size() != b.size() || a.empty()) {
        return s;
    }
    double sum = 0.0;
    for (size_t i = 0; i < a.size(); ++i) {
        const float d = std::fabs(a[i] - b[i]);
        s.max_abs = std::max(s.max_abs, d);
        sum += d;
    }
    s.mean_abs = (float) (sum / (double) a.size());
    return s;
}

int main(int argc, char ** argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: %s MODEL.gguf\n", argv[0]);
        return 1;
    }

#if defined(_WIN32)
    fprintf(stderr, "test-split-tcp: requires fork(); run split_sender and split_receiver manually on Windows\n");
    return 1;
#else
    const char * model_path = argv[1];
    const int port = 18000 + (getpid() % 1000);

    char out_tpl[] = "/tmp/llama_split_tcp_XXXXXX";
    const int out_fd = mkstemp(out_tpl);
    if (out_fd < 0) {
        fprintf(stderr, "failed to create temp file\n");
        return 1;
    }
    close(out_fd);

    const std::string dir         = exe_dir(argv[0]);
    const std::string recv_bin    = dir + "split_receiver";
    const std::string send_bin    = dir + "split_sender";
    const std::string port_str    = std::to_string(port);
    const std::string layer_end_s = "8"; // overridden after model load if needed

    ggml_backend_load_all();

    llama_model * model = llama_model_load_from_file(model_path, llama_model_default_params());
    if (!model) {
        fprintf(stderr, "failed to load model\n");
        return 1;
    }

    const int32_t n_layer = llama_model_n_layer(model);
    const int32_t half    = n_layer / 2;
    const int32_t n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model));
    const std::string layer_end_str = std::to_string(half);

    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx   = 512;
    cparams.no_perf = true;
    llama_context * ctx = llama_init_from_model(model, cparams);
    if (!ctx) {
        fprintf(stderr, "failed to create context\n");
        return 1;
    }

    const llama_vocab * vocab = llama_model_get_vocab(model);
    const std::string prompt = "Hello";
    const int n_tokens = -llama_tokenize(vocab, prompt.c_str(), prompt.size(), nullptr, 0, true, true);
    std::vector<llama_token> tokens(n_tokens);
    llama_tokenize(vocab, prompt.c_str(), prompt.size(), tokens.data(), tokens.size(), true, true);

    printf("model: n_layer=%d half=%d n_vocab=%d n_tokens=%d port=%d\n",
            n_layer, half, n_vocab, n_tokens, port);

    // full model baseline
    const int64_t t_full0 = ggml_time_us();
    llama_set_layer_range(ctx, 0, -1);
    llama_clear_hidden_state(ctx);
    llama_memory_clear(llama_get_memory(ctx), true);
    if (decode_tokens(ctx, tokens) != 0) {
        fprintf(stderr, "full forward failed\n");
        return 1;
    }
    const auto full_logits = get_last_logits(ctx, n_vocab);
    const double ms_full = (ggml_time_us() - t_full0) / 1000.0;
    printf("full forward: ms=%.3f logits[0..3]=%f %f %f %f\n",
            ms_full, full_logits[0], full_logits[1], full_logits[2], full_logits[3]);

    llama_free(ctx);
    llama_model_free(model);

    // Process B: receiver (listen, then load model, then accept)
    const pid_t pid_recv = fork();
    if (pid_recv == 0) {
        execl(recv_bin.c_str(), "split_receiver", model_path,
                "--port", port_str.c_str(),
                "--out", out_tpl,
                (char *) nullptr);
        fprintf(stderr, "execl receiver failed\n");
        _exit(127);
    }

    // allow receiver to bind before sender connects
    usleep(200000);

    // Process A: sender (connect)
    const int64_t t_split0 = ggml_time_us();
    const pid_t pid_send = fork();
    if (pid_send == 0) {
        execl(send_bin.c_str(), "split_sender", model_path,
                "--host", "127.0.0.1",
                "--port", port_str.c_str(),
                "--prompt", "Hello",
                "--layer-end", layer_end_str.c_str(),
                (char *) nullptr);
        fprintf(stderr, "execl sender failed\n");
        _exit(127);
    }

    int status = 0;
    waitpid(pid_send, &status, 0);
    if (!WIFEXITED(status) || WEXITSTATUS(status) != 0) {
        fprintf(stderr, "sender exited abnormally\n");
        kill(pid_recv, SIGTERM);
        waitpid(pid_recv, nullptr, 0);
        unlink(out_tpl);
        return 1;
    }

    waitpid(pid_recv, &status, 0);
    if (!WIFEXITED(status) || WEXITSTATUS(status) != 0) {
        fprintf(stderr, "receiver exited abnormally\n");
        unlink(out_tpl);
        return 1;
    }
    const double ms_split = (ggml_time_us() - t_split0) / 1000.0;

    split_tcp_result_file meta{};
    std::vector<float> split_logits;
    if (!split_tcp_read_result(out_tpl, meta, split_logits)) {
        fprintf(stderr, "failed to read receiver output\n");
        unlink(out_tpl);
        return 1;
    }
    unlink(out_tpl);

    printf("split TCP: wall_ms=%.3f recv_ms=%.3f decode_ms=%.3f\n",
            ms_split, meta.ms_recv, meta.ms_decode);
    printf("split logits[0..3]=%f %f %f %f\n",
            split_logits[0], split_logits[1], split_logits[2], split_logits[3]);

    const diff_stats diff = compare_vectors(full_logits, split_logits);
    printf("comparison: max_abs_diff=%.9f mean_abs_diff=%.9f\n", diff.max_abs, diff.mean_abs);
    printf("perf: full_ms=%.3f split_wall_ms=%.3f overhead_ms=%.3f\n",
            ms_full, ms_split, ms_split - meta.ms_decode - meta.ms_recv);

    const float tol = 1e-5f;
    if (diff.max_abs < tol) {
        printf("test-split-tcp: OK\n");
        return 0;
    }

    fprintf(stderr, "test-split-tcp: FAILED (max_abs_diff=%.9f)\n", diff.max_abs);
    return 1;
#endif
}
