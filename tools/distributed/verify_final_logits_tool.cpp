#include "verification/final_runtime_verify.h"

#include <cstdio>
#include <string>

static void usage(const char * prog) {
    fprintf(stderr,
            "usage: %s MONO_GGUF WORKER_FINAL_GGUF HIDDEN_BIN PROMPT [LAYER_BOUNDARY]\n",
            prog);
}

int main(int argc, char ** argv) {
    if (argc < 5) {
        usage(argv[0]);
        return 2;
    }

    final_logits_config cfg{};
    cfg.mono_path         = argv[1];
    cfg.worker_final_path = argv[2];
    cfg.hidden_bin_path   = argv[3];
    cfg.prompt            = argv[4];
    cfg.layer_boundary    = argc >= 6 ? std::stoi(argv[5]) : -1;

    const final_logits_report report = verify_final_logits(cfg);
    printf("%s\n", final_logits_report_json(report).c_str());
    return report.pass ? 0 : 1;
}
