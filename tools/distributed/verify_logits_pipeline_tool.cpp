#include "verification/logits_compare.h"
#include "verification/decode_loop_parity.h"

#include "nlohmann/json.hpp"

#include <cstdio>
#include <string>

using json = nlohmann::json;

int main(int argc, char ** argv) {
    if (argc < 4) {
        fprintf(stderr, "usage: %s MONO.gguf FINAL_TRACE.jsonl PROMPT\n", argv[0]);
        return 2;
    }
    const std::string mono_gguf  = argv[1];
    const std::string final_trace = argv[2];
    const std::string prompt       = argv[3];

    const auto cmp = compare_logits_files(mono_gguf, mono_gguf, prompt);
    const auto trace = parse_trace_jsonl(final_trace);

    const trace_event * final_logits = nullptr;
    for (const auto & ev : trace.events) {
        if (ev.event == "logits" && ev.phase == "prefill") {
            final_logits = &ev;
            break;
        }
    }

    json out;
    out["mono_logits_compare"] = cmp.summary.to_json();
    if (final_logits) {
        out["dist_prefill_logits"] = {
            { "argmax", final_logits->argmax },
            { "sha256", final_logits->logits_stats.sha256 },
            { "entropy", final_logits->entropy },
        };
        out["match"] = cmp.original.match;
    } else {
        out["error"] = "final trace missing prefill logits";
        out["match"] = false;
    }
    printf("%s\n", out.dump(2).c_str());
    return out.value("match", false) ? 0 : 1;
}
