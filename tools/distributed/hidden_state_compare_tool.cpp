#include "verification/hidden_state_verify.h"

#include <cstdio>

int main(int argc, char ** argv) {
    if (argc < 3) {
        fprintf(stderr, "usage: %s ORIGINAL.gguf MATERIALIZED.gguf [prompt]\n", argv[0]);
        return 1;
    }
    const std::string prompt = argc >= 4 ? argv[3] : "The capital of France is";
    const auto result        = compare_hidden_states(argv[1], argv[2], prompt);

    nlohmann::json out = result.summary.to_json();
    out["original"] = {
        { "sha256", result.original.sha256 },
        { "mean_abs", result.original.mean_abs },
        { "max_abs", result.original.max_abs },
    };
    out["materialized"] = {
        { "sha256", result.materialized.sha256 },
        { "mean_abs", result.materialized.mean_abs },
        { "max_abs", result.materialized.max_abs },
    };
    out["mae"]       = result.original.mae_vs_other;
    out["max_error"] = result.original.max_err;
    out["match"]     = result.original.match;
    printf("%s\n", out.dump(2).c_str());
    return result.summary.status == verify_status::ok ? 0 : 1;
}
