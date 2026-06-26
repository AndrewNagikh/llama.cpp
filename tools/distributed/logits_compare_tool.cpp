#include "verification/logits_compare.h"

#include <cstdio>

int main(int argc, char ** argv) {
    if (argc < 3) {
        fprintf(stderr, "usage: %s ORIGINAL.gguf MATERIALIZED.gguf [prompt]\n", argv[0]);
        return 1;
    }
    const std::string prompt = argc >= 4 ? argv[3] : "The capital of France is";
    const auto result        = compare_logits_files(argv[1], argv[2], prompt);

    nlohmann::json out = result.summary.to_json();
    out["original"] = {
        { "max_value", result.original.max_value },
        { "argmax", result.original.max_index },
    };
    out["materialized"] = {
        { "max_value", result.materialized.max_value },
        { "argmax", result.materialized.max_index },
    };
    out["mae"]        = result.original.mean_abs_err;
    out["max_error"]  = result.original.max_abs_err;
    out["match"]      = result.original.match;
    printf("%s\n", out.dump(2).c_str());
    return result.summary.status == verify_status::ok ? 0 : 1;
}
