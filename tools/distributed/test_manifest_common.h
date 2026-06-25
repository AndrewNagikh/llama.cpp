#pragma once

#include <cstdlib>
#include <filesystem>
#include <string>
#include <vector>

inline std::string manifest_test_model_path() {
    std::vector<std::string> candidates;
    if (const char * env = std::getenv("MODEL")) {
        candidates.push_back(env);
    }
    if (const char * home = std::getenv("HOME")) {
        candidates.push_back(std::string(home) + "/models/llama-3.2-1b-instruct-q4_k_m.gguf");
    }
    candidates.push_back("llama-3.2-1b-instruct-q4_k_m.gguf");

    for (const auto & c : candidates) {
        std::error_code ec;
        if (!c.empty() && std::filesystem::exists(c, ec)) {
            return c;
        }
    }
    return {};
}
