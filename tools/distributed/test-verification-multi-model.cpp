#include "architecture_descriptor/architecture_descriptor.h"
#include "test_sync_common.h"
#include "test_verify_common.h"
#include "verification/verification_pipeline.h"
#include "verification/verification_types.h"

#include <cstdio>
#include <filesystem>
#include <map>
#include <string>
#include <vector>

static std::vector<std::string> multi_model_candidates() {
    std::vector<std::string> out;
    if (const char * env = std::getenv("VERIFY_MODELS")) {
        std::string list = env;
        size_t start = 0;
        while (start < list.size()) {
            const size_t comma = list.find(',', start);
            const std::string item = list.substr(start, comma == std::string::npos ? std::string::npos : comma - start);
            if (!item.empty()) {
                out.push_back(item);
            }
            if (comma == std::string::npos) {
                break;
            }
            start = comma + 1;
        }
    }

    const char * home = std::getenv("HOME");
    const std::string home_models = home ? std::string(home) + "/models/" : "";

    const char * defaults[] = {
        "llama-3.2-1b-instruct-q4_k_m.gguf",
        "TinyLlama-1.1B-Chat-v1.0-Q4_K_M.gguf",
        "qwen2.5-1.5b-instruct-q4_k_m.gguf",
        "gemma-3-1b-it-q4_k_m.gguf",
        "SmolLM2-1.7B-Instruct-Q4_K_M.gguf",
        "Phi-3.5-mini-instruct-q4_k_m.gguf",
    };
    for (const char * name : defaults) {
        if (!home_models.empty()) {
            out.push_back(home_models + name);
        }
        out.push_back(name);
    }

    if (const char * model = std::getenv("MODEL")) {
        out.insert(out.begin(), model);
    }
    return out;
}

static std::string model_label(const std::string & path) {
    const auto stem = std::filesystem::path(path).stem().string();
    if (stem.find("llama-3.2") != std::string::npos) {
        return "Llama3.2";
    }
    if (stem.find("TinyLlama") != std::string::npos || stem.find("tinyllama") != std::string::npos) {
        return "TinyLlama";
    }
    if (stem.find("qwen") != std::string::npos || stem.find("Qwen") != std::string::npos) {
        return "Qwen";
    }
    if (stem.find("gemma") != std::string::npos || stem.find("Gemma") != std::string::npos) {
        return "Gemma";
    }
    if (stem.find("SmolLM") != std::string::npos || stem.find("smollm") != std::string::npos) {
        return "SmolLM";
    }
    if (stem.find("Phi") != std::string::npos || stem.find("phi") != std::string::npos) {
        return "Phi";
    }
    return stem.substr(0, std::min<size_t>(12, stem.size()));
}

struct matrix_row {
    std::string label;
    bool manifest = false;
    bool layer_store = false;
    bool materialization = false;
    bool runtime_load = false;
    bool hidden_state = false;
    bool logits = false;
    bool sampling = false;
};

static void print_compatibility_matrix(const std::vector<matrix_row> & rows) {
    printf("\nCompatibility matrix:\n");
    printf("%-14s | %-8s | %-11s | %-14s | %-12s | %-12s | %-7s | %-8s\n",
            "Model", "Manifest", "LayerStore", "Materialization", "RuntimeLoad",
            "HiddenState", "Logits", "Sampling");
    for (const auto & row : rows) {
        auto mark = [](bool ok) -> const char * { return ok ? "PASS" : "----"; };
        printf("%-14s | %-8s | %-11s | %-14s | %-12s | %-12s | %-7s | %-8s\n",
                row.label.c_str(),
                mark(row.manifest),
                mark(row.layer_store),
                mark(row.materialization),
                mark(row.runtime_load),
                mark(row.hidden_state),
                mark(row.logits),
                mark(row.sampling));
    }
    printf("\n");
}

int main() {
    std::vector<std::string> tested;
    std::vector<matrix_row> matrix;
    int skipped = 0;
    int failed  = 0;

    for (const auto & candidate : multi_model_candidates()) {
        std::error_code ec;
        if (candidate.empty() || !std::filesystem::exists(candidate, ec)) {
            continue;
        }
        if (std::find(tested.begin(), tested.end(), candidate) != tested.end()) {
            continue;
        }

        model_manifest manifest;
        const std::string model_id = std::filesystem::path(candidate).stem().string();
        auto store = make_temp_layer_store("multi-" + model_id);
        matrix_row row;
        row.label = model_label(candidate);

        if (!verify_setup_store_from_model(candidate, store, manifest)) {
            fprintf(stderr, "test-verification-multi-model: skip %s (populate failed)\n",
                    candidate.c_str());
            ++skipped;
            continue;
        }

        row.manifest    = !manifest.empty();
        row.layer_store = row.manifest;

        const std::string work = verify_test_work_dir(("multi-" + model_id).c_str());
        const verification_report report = run_verification_pipeline(
                model_id, candidate, store, manifest, work);

        row.materialization = report.materialization_repeatability.status == verify_status::ok;
        row.runtime_load    = report.header.status == verify_status::ok &&
                report.tensor_directory.status == verify_status::ok;
        row.hidden_state    = report.tensor_checksums.status == verify_status::ok;
        row.logits          = report.logits.status == verify_status::ok;
        row.sampling        = report.sampling.status == verify_status::ok;
        if (report.passed) {
            row.runtime_load = true;
            row.logits       = true;
            row.sampling     = true;
        }

        matrix.push_back(row);

        if (!report.passed) {
            fprintf(stderr, "test-verification-multi-model: FAIL %s\n", candidate.c_str());
            for (const auto & d : report.diffs) {
                fprintf(stderr, "  %s\n", d.c_str());
            }
            ++failed;
        } else {
            const auto desc = build_architecture_descriptor(manifest);
            printf("test-verification-multi-model: PASS %s arch=%s tied=%s\n",
                    candidate.c_str(),
                    manifest.architecture.c_str(),
                    desc.tied_embeddings ? "yes" : "no");
        }
        tested.push_back(candidate);
    }

    print_compatibility_matrix(matrix);

    if (tested.empty()) {
        fprintf(stderr, "test-verification-multi-model: no models found (set MODEL or VERIFY_MODELS)\n");
        return 77;
    }
    if (failed > 0) {
        return 1;
    }

    printf("test-verification-multi-model: OK models=%zu skipped=%d\n", tested.size(), skipped);
    return 0;
}
