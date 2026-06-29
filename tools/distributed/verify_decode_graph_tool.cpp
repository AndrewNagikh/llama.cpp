#include "verification/decode_graph_verify.h"
#include "verification/final_runtime_verify.h"

#include "split_gen_common.h"

#include "ggml-backend.h"
#include "llama.h"

#include <cstdio>
#include <filesystem>
#include <fstream>
#include <string>

static void usage(const char * prog) {
    fprintf(stderr,
            "usage: %s MONO_GGUF WORKER_FINAL_GGUF PROMPT LAYER_BOUNDARY [--out DIR] [--prefill]\n",
            prog);
}

int main(int argc, char ** argv) {
    if (argc < 5) {
        usage(argv[0]);
        return 2;
    }

    const std::string mono_path   = argv[1];
    const std::string worker_path = argv[2];
    const std::string prompt      = argv[3];
    const int32_t layer_boundary  = std::stoi(argv[4]);
    std::string output_dir        = "./logs";
    bool prefill                  = false;

    for (int i = 5; i < argc; ++i) {
        if (std::string(argv[i]) == "--out" && i + 1 < argc) {
            output_dir = argv[++i];
        } else if (std::string(argv[i]) == "--prefill") {
            prefill = true;
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

    const int32_t n_layer = llama_model_n_layer(mono_model);
    const auto tokens     = split_gen_tokenize(llama_model_get_vocab(mono_model), prompt);
    const uint32_t n_tok  = prefill ? static_cast<uint32_t>(tokens.size()) : 1u;

    const decode_graph_verify_result result = verify_decode_graph(
            mono_ctx,
            worker_ctx,
            layer_boundary,
            n_layer,
            n_tok,
            prefill ? "prefill" : "decode");

    const std::string json = decode_graph_diff_json(result);
    printf("%s\n", json.c_str());

    std::error_code ec;
    std::filesystem::create_directories(output_dir, ec);
    write_graph_summary_json(output_dir + "/decode_graph_monolithic.json", result.mono);
    write_graph_summary_json(output_dir + "/decode_graph_worker.json", result.worker);

    std::ofstream diff_out(output_dir + "/decode_graph_diff.json");
    if (diff_out) {
        diff_out << json;
    }

    llama_free(mono_ctx);
    llama_free(worker_ctx);
    llama_model_free(mono_model);
    llama_model_free(worker_model);
    return result.comparison.match ? 0 : 1;
}
