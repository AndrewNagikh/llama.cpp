#include "verification/tensor_verify.h"

#include <cstdio>

int main(int argc, char ** argv) {
    if (argc < 3) {
        fprintf(stderr, "usage: %s ORIGINAL.gguf MATERIALIZED.gguf\n", argv[0]);
        return 1;
    }
    const auto result = tensor_verify_files(argv[1], argv[2]);
    nlohmann::json out = result.summary.to_json();
    out["tensors"] = nlohmann::json::array();
    for (const auto & t : result.tensors) {
        out["tensors"].push_back({
            { "name", t.name },
            { "offset", t.offset },
            { "size_bytes", t.size_bytes },
            { "ggml_type", t.ggml_type },
            { "sha256_original", t.sha256_original },
            { "sha256_materialized", t.sha256_materialized },
            { "match", t.match },
        });
    }
    out["mismatches"] = result.mismatches;
    printf("%s\n", out.dump(2).c_str());
    return result.summary.status == verify_status::ok ? 0 : 1;
}
