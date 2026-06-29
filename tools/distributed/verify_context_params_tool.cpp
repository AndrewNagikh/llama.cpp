#include "verification/context_params_verify.h"

#include "split_gen_common.h"

#include "ggml-backend.h"
#include "llama.h"

#include <cstdio>
#include <filesystem>
#include <fstream>
#include <string>

static void usage(const char * prog) {
    fprintf(stderr, "usage: %s MONO_GGUF WORKER_FINAL_GGUF PROMPT [--out DIR]\n", prog);
}

int main(int argc, char ** argv) {
    if (argc < 4) {
        usage(argv[0]);
        return 2;
    }

    const std::string mono_path   = argv[1];
    const std::string worker_path = argv[2];
    const std::string prompt      = argv[3];
    std::string output_dir        = "./logs";

    for (int i = 4; i < argc; ++i) {
        if (std::string(argv[i]) == "--out" && i + 1 < argc) {
            output_dir = argv[++i];
        }
    }

    ggml_backend_load_all();

    llama_model * mono_model = llama_model_load_from_file(mono_path.c_str(), llama_model_default_params());
    llama_model * worker_model = llama_model_load_from_file(worker_path.c_str(), llama_model_default_params());
    if (!mono_model || !worker_model) {
        fprintf(stderr, "model load failed\n");
        return 1;
    }

    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx   = 512;
    cparams.n_batch = 512;
    cparams.no_perf = true;

    llama_context * mono_ctx   = llama_init_from_model(mono_model, cparams);
    llama_context * worker_ctx = llama_init_from_model(worker_model, cparams);
    if (!mono_ctx || !worker_ctx) {
        fprintf(stderr, "context init failed\n");
        return 1;
    }

    const auto tokens = split_gen_tokenize(llama_model_get_vocab(mono_model), prompt);
    if (!tokens.empty()) {
        split_gen_decode_tokens(mono_ctx, tokens, 0, false);
        split_gen_decode_tokens(worker_ctx, tokens, 0, false);
    }

    const context_params_diff_report report = compare_context_params(mono_ctx, worker_ctx);
    const std::string json                  = context_params_diff_json(report);
    printf("%s\n", json.c_str());

    std::error_code ec;
    std::filesystem::create_directories(output_dir, ec);
    std::ofstream out(output_dir + "/context_params_diff.json");
    if (out) {
        out << json;
    }

    llama_free(mono_ctx);
    llama_free(worker_ctx);
    llama_model_free(mono_model);
    llama_model_free(worker_model);
    return report.match ? 0 : 1;
}
