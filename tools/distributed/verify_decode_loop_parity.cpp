#include "verification/decode_loop_parity.h"
#include "verification/monolithic_trace.h"

#include "runtime_debug/runtime_debug.h"

#include "nlohmann/json.hpp"

#include <cstdio>
#include <filesystem>
#include <string>
#include <vector>

using json = nlohmann::json;

static void usage(const char * prog) {
    fprintf(stderr,
            "usage: %s --mono-trace PATH --dist-entry PATH [--dist-middle PATH] [--dist-final PATH]\n"
            "       %s --model PATH --prompt TEXT [--dist-dir DIR]\n",
            prog, prog);
}

static int analyze_traces(
        const parsed_trace & mono,
        const parsed_trace & entry,
        const parsed_trace & middle,
        const parsed_trace & final) {
    const divergence_report div = find_first_divergence(mono, entry, final, middle);
    json out;
    if (div.kind == divergence_kind::none) {
        out["status"]  = "ok";
        out["message"] = "no divergence detected in compared events";
        printf("%s\n", out.dump(2).c_str());
        return 0;
    }
    out["status"]  = "divergence";
    out["kind"]    = divergence_kind_name(div.kind);
    out["step"]    = div.step;
    out["phase"]   = div.phase;
    out["worker"]  = div.worker;
    out["field"]   = div.field;
    out["message"] = div.message;
    out["root_cause"] = div.root_cause_hint;
    out["mono_source"] = div.mono_source;
    out["dist_source"] = div.dist_source;
    printf("%s\n", out.dump(2).c_str());
    return 1;
}

int main(int argc, char ** argv) {
    std::string mono_trace;
    std::string dist_entry;
    std::string dist_middle;
    std::string dist_final;
    std::string model_path;
    std::string prompt = "The capital of France is";
    std::string dist_dir;

    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        auto next = [&](std::string & out) {
            if (i + 1 < argc) {
                out = argv[++i];
            }
        };
        if (arg == "--mono-trace") {
            next(mono_trace);
        } else if (arg == "--dist-entry") {
            next(dist_entry);
        } else if (arg == "--dist-middle") {
            next(dist_middle);
        } else if (arg == "--dist-final") {
            next(dist_final);
        } else if (arg == "--model") {
            next(model_path);
        } else if (arg == "--prompt") {
            next(prompt);
        } else if (arg == "--dist-dir") {
            next(dist_dir);
        } else {
            usage(argv[0]);
            return 2;
        }
    }

    if (!model_path.empty()) {
        setenv("LLAMA_DISTRIBUTED_DEBUG", "1", 1);
        const mono_trace_result mono = run_monolithic_trace(model_path, prompt, 16, "mono_ref");
        if (!mono.ok) {
            fprintf(stderr, "mono trace failed: %s\n", mono.error.c_str());
            return 1;
        }
        mono_trace = mono.trace_path;
        if (!dist_dir.empty()) {
            namespace fs = std::filesystem;
            for (const auto & ent : fs::directory_iterator(dist_dir)) {
                if (!ent.is_regular_file()) {
                    continue;
                }
                const std::string name = ent.path().filename().string();
                if (name.find("_entry") != std::string::npos) {
                    dist_entry = ent.path().string();
                } else if (name.find("_middle") != std::string::npos) {
                    dist_middle = ent.path().string();
                } else if (name.find("_final") != std::string::npos) {
                    dist_final = ent.path().string();
                }
            }
        }
    }

    if (mono_trace.empty() || dist_entry.empty()) {
        usage(argv[0]);
        return 2;
    }

    return analyze_traces(
            parse_trace_jsonl(mono_trace),
            parse_trace_jsonl(dist_entry),
            parse_trace_jsonl(dist_middle),
            parse_trace_jsonl(dist_final));
}
