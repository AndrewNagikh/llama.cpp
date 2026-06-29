#include "architecture/architecture_plugin.h"
#include "orchestrator/manifest_builder/manifest_builder.h"
#include "runtime/architecture_descriptor/distributed_runtime_descriptor.h"
#include "verification/verification_suite.h"

#include "nlohmann/json.hpp"

#include <cstdio>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

namespace {

struct matrix_entry {
    std::string model_id;
    std::string family;
    std::string gguf_path;
};

std::vector<matrix_entry> load_matrix(const std::string & path) {
    std::vector<matrix_entry> out;
    std::ifstream in(path);
    if (!in) {
        return out;
    }
    try {
        const nlohmann::json j = nlohmann::json::parse(in);
        for (const auto & item : j.value("families", nlohmann::json::array())) {
            const std::string family = item.value("family", "");
            for (const auto & model : item.value("models", nlohmann::json::array())) {
                matrix_entry e;
                e.model_id  = model.value("model_id", "");
                e.family    = family;
                e.gguf_path = model.value("gguf_path", "");
                if (!e.model_id.empty()) {
                    out.push_back(std::move(e));
                }
            }
        }
    } catch (...) {
    }
    return out;
}

} // namespace

int main(int argc, char ** argv) {
    std::string matrix_path = "config/architecture_matrix.json";
    std::string output_path = "logs/architecture_report.json";
    std::string single_gguf;

    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--matrix" && i + 1 < argc) {
            matrix_path = argv[++i];
        } else if (arg == "--output" && i + 1 < argc) {
            output_path = argv[++i];
        } else if (arg == "--gguf" && i + 1 < argc) {
            single_gguf = argv[++i];
        }
    }

    nlohmann::json report = nlohmann::json::array();

    if (!single_gguf.empty()) {
        model_manifest manifest = build_manifest_from_file(single_gguf);
        if (manifest.empty()) {
            fprintf(stderr, "architecture-report: failed to build manifest from %s\n",
                    single_gguf.c_str());
            return 1;
        }
        const architecture_plugin & plugin = select_architecture_plugin(manifest);
        const distributed_runtime_descriptor desc = plugin.build_distributed_descriptor(manifest);
        std::string verr;
        const bool runtime_ok = plugin.verify_runtime(desc, verr);
        const auto suite = run_local_verification_suite(
                std::filesystem::path(single_gguf).stem().string(),
                manifest,
                single_gguf);
        report.push_back({
            { "family", desc.semantic.family },
            { "model_id", std::filesystem::path(single_gguf).stem().string() },
            { "gguf_path", single_gguf },
            { "descriptor", desc.to_json() },
            { "runtime_status", runtime_ok ? "ok" : "error" },
            { "runtime_notes", verr },
            { "tests_passed", suite.all_passed() },
            { "verification", suite.to_json() },
            { "partial_forward", desc.capabilities.supports_partial_forward },
            { "hidden_injection", desc.capabilities.supports_hidden_injection },
        });
    } else {
        for (const matrix_entry & entry : load_matrix(matrix_path)) {
            nlohmann::json item = {
                { "family", entry.family },
                { "model_id", entry.model_id },
                { "gguf_path", entry.gguf_path },
            };
            if (!entry.gguf_path.empty() && std::filesystem::exists(entry.gguf_path)) {
                model_manifest manifest = build_manifest_from_file(entry.gguf_path);
                if (!manifest.empty()) {
                    const architecture_plugin & plugin = select_architecture_plugin(manifest);
                    const distributed_runtime_descriptor desc =
                            plugin.build_distributed_descriptor(manifest);
                    std::string verr;
                    const bool runtime_ok = plugin.verify_runtime(desc, verr);
                    const auto suite = run_local_verification_suite(
                            entry.model_id, manifest, entry.gguf_path);
                    item["descriptor"]      = desc.to_json();
                    item["runtime_status"]  = runtime_ok ? "ok" : "error";
                    item["runtime_notes"]   = verr;
                    item["tests_passed"]    = suite.all_passed();
                    item["verification"]    = suite.to_json();
                    item["partial_forward"] = desc.capabilities.supports_partial_forward;
                    item["hidden_injection"] = desc.capabilities.supports_hidden_injection;
                } else {
                    item["runtime_status"] = "manifest_error";
                    item["runtime_notes"]  = "build_manifest_from_file returned empty manifest";
                    item["tests_passed"]   = false;
                }
            } else {
                item["runtime_status"] = "no_gguf";
                item["tests_passed"]   = false;
                item["notes"]          = "GGUF not available locally — run cluster E2E";
            }
            report.push_back(item);
        }
    }

    std::error_code ec;
    std::filesystem::create_directories(std::filesystem::path(output_path).parent_path(), ec);
    std::ofstream out(output_path);
    if (!out) {
        fprintf(stderr, "architecture-report: cannot write %s\n", output_path.c_str());
        return 1;
    }
    out << report.dump(2) << '\n';
    printf("architecture-report: wrote %s (%zu families/models)\n",
            output_path.c_str(), report.size());
    return 0;
}
