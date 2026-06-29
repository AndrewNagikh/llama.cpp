#include "verification/decode_loop_parity.h"

#include <cassert>
#include <cstdio>
#include <filesystem>
#include <fstream>

static void write_line(const std::string & path, const std::string & line) {
    std::ofstream out(path, std::ios::app);
    out << line << '\n';
}

int main() {
    const std::string dir = "/tmp/decode_parity_test";
    std::filesystem::create_directories(dir);
    const std::string mono = dir + "/mono.jsonl";
    const std::string dist = dir + "/entry.jsonl";
    std::filesystem::remove(mono);
    std::filesystem::remove(dist);

    write_line(mono,
            R"({"event":"token_selected","step":0,"phase":"prefill","worker":"monolithic","session":"s","node":"local","token":100,"position":5,"ts_ms":1})");
    write_line(dist,
            R"({"event":"token_selected","step":0,"phase":"prefill","worker":"final","session":"s","node":"node-c","token":100,"position":5,"ts_ms":2})");

    parsed_trace mono_t = parse_trace_jsonl(mono);
    parsed_trace dist_t = parse_trace_jsonl(dist);
    parsed_trace empty{};

    divergence_report ok = find_first_divergence(mono_t, dist_t, dist_t, empty);
    assert(ok.kind == divergence_kind::none);

    write_line(dist,
            R"({"event":"token_selected","step":0,"phase":"prefill","worker":"final","session":"s","node":"node-c","token":999,"position":5,"ts_ms":3})");
    dist_t = parse_trace_jsonl(dist);
    divergence_report bad = find_first_divergence(mono_t, dist_t, dist_t, empty);
    assert(bad.kind == divergence_kind::selected_token);
    assert(bad.step == 0);

    printf("test-decode-loop-parity: OK root_cause=%s\n", bad.root_cause_hint.c_str());
    return 0;
}
