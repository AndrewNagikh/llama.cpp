#include "verification/final_runtime_verify.h"

#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <string>

static void usage(const char * prog) {
    fprintf(stderr,
            "usage: %s MONO_GGUF WORKER_FINAL_GGUF PROMPT [LAYER_BOUNDARY] [--out DIR] [--decode-steps N]\n"
            "\n"
            "Task 9.8.6 — isolate Final Runtime from Entry/Middle/TCP.\n"
            "Writes runtime_equivalence.json to --out (default: ./logs).\n",
            prog);
}

int main(int argc, char ** argv) {
    if (argc < 4) {
        usage(argv[0]);
        return 2;
    }

    final_runtime_config cfg{};
    cfg.mono_path         = argv[1];
    cfg.worker_final_path = argv[2];
    cfg.prompt            = argv[3];
    cfg.layer_boundary    = argc >= 5 && argv[4][0] != '-' ? std::stoi(argv[4]) : -1;
    cfg.output_dir        = "./logs";
    cfg.max_decode_steps  = 8;

    for (int i = 4; i < argc; ++i) {
        if (std::string(argv[i]) == "--out" && i + 1 < argc) {
            cfg.output_dir = argv[++i];
        } else if (std::string(argv[i]) == "--decode-steps" && i + 1 < argc) {
            cfg.max_decode_steps = std::stoi(argv[++i]);
        }
    }

    const final_runtime_report report = verify_final_runtime(cfg);
    const std::string json            = final_runtime_report_json(report);

    printf("%s\n", json.c_str());

    std::error_code ec;
    std::filesystem::create_directories(cfg.output_dir, ec);
    std::ofstream out(cfg.output_dir + "/runtime_equivalence.json");
    if (out) {
        out << json;
    }

    return report.all_pass ? 0 : 1;
}
