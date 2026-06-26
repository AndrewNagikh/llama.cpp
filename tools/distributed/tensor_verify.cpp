#include "verification/tensor_verify.h"
#include "verification/worker_tensor_plan.h"

#include "orchestrator/manifest_builder/manifest_builder.h"

#include <cstdio>
#include <cstring>

static void usage(const char * prog) {
    fprintf(stderr,
            "usage: %s ORIGINAL.gguf MATERIALIZED.gguf\n"
            "       %s --entry LAYER_END ORIGINAL.gguf MATERIALIZED.gguf\n"
            "       %s --final LAYER_START LAYER_END ORIGINAL.gguf MATERIALIZED.gguf\n"
            "\n"
            "Full compare checks every tensor. Worker modes check only tensors\n"
            "owned by ENTRY/FINAL partial GGUF (excluded regions may be zero).\n",
            prog, prog, prog);
}

int main(int argc, char ** argv) {
    worker_tensor_plan plan;
    bool worker_mode = false;
    int argi = 1;

    if (argc >= 2 && strcmp(argv[1], "--entry") == 0) {
        if (argc < 5) {
            usage(argv[0]);
            return 1;
        }
        plan = make_worker_tensor_plan(
                worker_verify_role::entry,
                0,
                static_cast<int32_t>(std::stoi(argv[2])));
        worker_mode = true;
        argi = 3;
    } else if (argc >= 2 && strcmp(argv[1], "--final") == 0) {
        if (argc < 6) {
            usage(argv[0]);
            return 1;
        }
        plan = make_worker_tensor_plan(
                worker_verify_role::final,
                static_cast<int32_t>(std::stoi(argv[2])),
                static_cast<int32_t>(std::stoi(argv[3])));
        worker_mode = true;
        argi = 4;
    }

    if (argc - argi < 2) {
        usage(argv[0]);
        return 1;
    }

    const std::string original     = argv[argi];
    const std::string materialized = argv[argi + 1];

    tensor_verify_result result;
    if (worker_mode) {
        const model_manifest manifest = build_manifest_from_file(original);
        if (manifest.empty()) {
            fprintf(stderr, "tensor_verify: failed to build manifest from %s\n", original.c_str());
            return 1;
        }
        result = tensor_verify_worker_files(original, materialized, manifest, plan);
    } else {
        result = tensor_verify_files(original, materialized);
    }

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
    out["skipped"]    = result.skipped;
    printf("%s\n", out.dump(2).c_str());
    return result.summary.status == verify_status::ok ? 0 : 1;
}
