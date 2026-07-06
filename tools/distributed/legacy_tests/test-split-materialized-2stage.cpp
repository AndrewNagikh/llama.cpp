// 2-stage split (entry + final) using materialized partial GGUFs from layer store.

#include "split_gen3_common.h"
#include "split_gen_common.h"
#include "split_tcp_wire.h"
#include "test_sync_common.h"
#include "test_verify_common.h"

#include "node_agent/layer_store/layer_gguf_assembler.h"

#include "ggml-backend.h"
#include "llama.h"

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

#if !defined(_WIN32)

static bool gen3_send_recv(
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

static std::vector<llama_token> run_split2(
        int ctrl_fd,
        const std::vector<llama_token> & prompt,
        int max_new,
        int layer_end) {
    std::vector<llama_token> out;

    split_gen3_a_resp resp{};
    if (!gen3_send_recv(ctrl_fd, SPLIT_GEN_CMD_RESET, 0, 0, layer_end, nullptr, resp)) {
        fprintf(stderr, "split2: reset failed\n");
        return out;
    }

    std::vector<int32_t> ptoks(prompt.size());
    for (size_t i = 0; i < prompt.size(); ++i) {
        ptoks[i] = (int32_t) prompt[i];
    }

    if (!gen3_send_recv(ctrl_fd, SPLIT_GEN_CMD_PREFILL, (int32_t) prompt.size(), 0,
            layer_end, ptoks.data(), resp)) {
        fprintf(stderr, "split2: prefill failed\n");
        return out;
    }

    out.push_back((llama_token) resp.token_id);
    llama_token cur = (llama_token) resp.token_id;

    const int n_prompt = (int) prompt.size();
    for (int step = 1; step < max_new; ++step) {
        const int32_t tok = (int32_t) cur;
        const int32_t pos = n_prompt + step - 1;
        if (!gen3_send_recv(ctrl_fd, SPLIT_GEN_CMD_DECODE, 1, pos, layer_end, &tok, resp)) {
            fprintf(stderr, "split2: decode failed step=%d\n", step);
            break;
        }
        cur = (llama_token) resp.token_id;
        out.push_back(cur);
    }

    gen3_send_recv(ctrl_fd, SPLIT_GEN_CMD_SHUTDOWN, 0, 0, layer_end, nullptr, resp);
    return out;
}

#endif

int main(int argc, char ** argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: %s MODEL.gguf [layer_end=8] [max_new=8]\n", argv[0]);
        return 1;
    }

#if defined(_WIN32)
    fprintf(stderr, "test-split-materialized-2stage requires fork() (Unix)\n");
    return 1;
#else
    const char * model_path = argv[1];
    const int layer_end     = argc >= 3 ? atoi(argv[2]) : 8;
    const int max_new       = argc >= 4 ? atoi(argv[3]) : 8;

    model_manifest manifest;
    auto store = make_temp_layer_store("split-mat-2stage");
    if (!verify_setup_store_from_model(model_path, store, manifest)) {
        fprintf(stderr, "failed to populate layer store\n");
        return 1;
    }

    const std::string work = verify_test_work_dir("split-mat-2stage");
    const std::string entry_gguf = work + "/worker_entry.gguf";
    const std::string final_gguf = work + "/worker_final.gguf";

    if (!layer_store_materialize_gguf(store, manifest, entry_gguf, 0, layer_end, true, false)) {
        fprintf(stderr, "failed to materialize entry GGUF [0,%d)\n", layer_end);
        return 1;
    }
    if (!layer_store_materialize_gguf(
                store, manifest, final_gguf, layer_end, (int32_t) manifest.n_layer, false, true)) {
        fprintf(stderr, "failed to materialize final GGUF [%d,%zu)\n", layer_end, manifest.n_layer);
        return 1;
    }

    printf("materialized entry=%s final=%s split=[0,%d)+[%d,%zu)\n",
            entry_gguf.c_str(), final_gguf.c_str(), layer_end, layer_end, manifest.n_layer);

    ggml_backend_load_all();
    llama_model * model = llama_model_load_from_file(model_path, llama_model_default_params());
    if (!model) {
        fprintf(stderr, "failed to load model for tokenize\n");
        return 1;
    }
    const auto prompt = split_gen_tokenize(llama_model_get_vocab(model), "The capital of France is");
    llama_model_free(model);
    if (prompt.empty()) {
        fprintf(stderr, "empty prompt tokens\n");
        return 1;
    }

    const std::string dir = exe_dir(argv[0]);
    const int bc_port   = 25400 + (getpid() % 500);
    const int ctrl_port = bc_port + 1;

    pid_t pid_c = fork();
    if (pid_c == 0) {
        const std::string bc = std::to_string(bc_port);
        const std::string ls = std::to_string(layer_end);
        execl((dir + "split_gen3_c").c_str(), "split_gen3_c", final_gguf.c_str(),
                "--bc-port", bc.c_str(), "--layer-start", ls.c_str(), (char *) nullptr);
        _exit(127);
    }

    usleep(500000);

    pid_t pid_a = fork();
    if (pid_a == 0) {
        const std::string ctrl = std::to_string(ctrl_port);
        const std::string bc   = std::to_string(bc_port);
        const std::string le   = std::to_string(layer_end);
        execl((dir + "split_gen3_a").c_str(), "split_gen3_a", entry_gguf.c_str(),
                "--ctrl-port", ctrl.c_str(), "--b-port", bc.c_str(), "--b-host", "127.0.0.1",
                "--layer-end", le.c_str(), "--next-final", (char *) nullptr);
        _exit(127);
    }

    int ctrl_fd = -1;
    for (int retry = 0; retry < 300; ++retry) {
        ctrl_fd = split_tcp_connect("127.0.0.1", ctrl_port);
        if (ctrl_fd >= 0) {
            break;
        }
        usleep(100000);
    }
    if (ctrl_fd < 0) {
        fprintf(stderr, "connect to entry worker failed\n");
        kill(pid_a, SIGTERM);
        kill(pid_c, SIGTERM);
        return 1;
    }

    const auto split_tokens = run_split2(ctrl_fd, prompt, max_new, layer_end);
    close(ctrl_fd);

    int st = 0;
    waitpid(pid_a, &st, 0);
    waitpid(pid_c, &st, 0);

    printf("split tokens (%zu):", split_tokens.size());
    for (const auto t : split_tokens) {
        printf(" %d", (int) t);
    }
    printf("\n");

    const bool all_zero = !split_tokens.empty();
    for (const auto t : split_tokens) {
        if (t != 0) {
            // not all zero
            if (all_zero) {
                // first non-zero found - ok
            }
        }
    }
    bool has_nonzero = false;
    for (const auto t : split_tokens) {
        if (t != 0) {
            has_nonzero = true;
            break;
        }
    }

    if (split_tokens.empty()) {
        fprintf(stderr, "test-split-materialized-2stage: FAIL (no tokens)\n");
        return 1;
    }
    if (!has_nonzero) {
        fprintf(stderr, "test-split-materialized-2stage: FAIL (all token 0)\n");
        return 1;
    }

    printf("test-split-materialized-2stage: OK\n");
    return 0;
#endif
}
