#include "test_sync_common.h"
#include "test_verify_common.h"
#include "verification/verification_pipeline.h"

#include <cstdio>
#include <filesystem>
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
        "qwen2.5-0.5b-instruct-q4_k_m.gguf",
        "gemma-2-2b-it-q4_k_m.gguf",
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

int main() {
    std::vector<std::string> tested;
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
        if (!verify_setup_store_from_model(candidate, store, manifest)) {
            fprintf(stderr, "test-verification-multi-model: skip %s (populate failed)\n",
                    candidate.c_str());
            ++skipped;
            continue;
        }

        const std::string work = verify_test_work_dir(("multi-" + model_id).c_str());
        const verification_report report = run_verification_pipeline(
                model_id, candidate, store, manifest, work);
        if (!report.passed) {
            fprintf(stderr, "test-verification-multi-model: FAIL %s\n", candidate.c_str());
            for (const auto & d : report.diffs) {
                fprintf(stderr, "  %s\n", d.c_str());
            }
            ++failed;
        } else {
            printf("test-verification-multi-model: PASS %s\n", candidate.c_str());
        }
        tested.push_back(candidate);
    }

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
