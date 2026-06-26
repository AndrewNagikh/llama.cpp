#pragma once

#include "architecture_descriptor/architecture_descriptor.h"
#include "orchestrator/manifest_builder/manifest_builder.h"

#include <cstdlib>
#include <filesystem>
#include <optional>
#include <string>
#include <vector>

inline tensor_descriptor make_tensor(
        const std::string & name,
        const tensor_role role,
        const int32_t layer = -1) {
    tensor_descriptor t;
    t.name = name;
    t.role = role;
    t.layer = layer;
    t.size_bytes = 1024;
    return t;
}

inline model_manifest make_dense_manifest(
        const std::string & architecture,
        const bool tied_embeddings,
        const uint32_t n_layer = 4) {
    model_manifest m;
    m.architecture = architecture;
    m.n_layer      = n_layer;
    m.tensors.push_back(make_tensor("token_embd.weight", tensor_role::embedding));
    m.special_tensors["embedding"] = "token_embd.weight";

    if (!tied_embeddings) {
        m.tensors.push_back(make_tensor("output.weight", tensor_role::lm_head));
        m.special_tensors["lm_head"] = "output.weight";
    }

    m.tensors.push_back(make_tensor("output_norm.weight", tensor_role::output_norm));
    m.special_tensors["output_norm"] = "output_norm.weight";

    for (uint32_t i = 0; i < n_layer; ++i) {
        const std::string prefix = "blk." + std::to_string(i) + ".";
        m.tensors.push_back(make_tensor(prefix + "attn_q.weight", tensor_role::layer, (int32_t) i));
        m.tensors.push_back(make_tensor(prefix + "ffn_up.weight", tensor_role::layer, (int32_t) i));
    }

    m.layers = build_layer_descriptors(m.tensors);
    return m;
}

inline std::optional<std::string> find_model_file(const std::vector<std::string> & candidates) {
    if (const char * env = std::getenv("MODEL")) {
        std::error_code ec;
        if (std::filesystem::exists(env, ec)) {
            return std::string(env);
        }
    }

    const char * home = std::getenv("HOME");
    const std::string home_models = home ? std::string(home) + "/models/" : "";

    for (const auto & candidate : candidates) {
        std::error_code ec;
        if (std::filesystem::exists(candidate, ec)) {
            return candidate;
        }
        if (!home_models.empty()) {
            const std::string path = home_models + candidate;
            if (std::filesystem::exists(path, ec)) {
                return path;
            }
        }
    }
    return std::nullopt;
}

inline bool test_descriptor_from_gguf(
        const std::vector<std::string> & candidates,
        const std::string & expected_arch_prefix) {
    const auto path = find_model_file(candidates);
    if (!path.has_value()) {
        return false;
    }

    const model_manifest manifest = build_manifest_from_file(*path);
    if (manifest.empty()) {
        return false;
    }

    const auto desc = build_architecture_descriptor(manifest);
    return desc.architecture.rfind(expected_arch_prefix, 0) == 0 && !desc.tensors.empty();
}
