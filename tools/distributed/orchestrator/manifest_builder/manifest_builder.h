#pragma once

#include "nlohmann/json.hpp"

#include <cstdint>
#include <map>
#include <string>
#include <vector>

struct dist_model_record;

// ---------------------------------------------------------------------------
// GGUF Manifest Builder - Task 9.3
//
// Builds a full model structure description from GGUF header, metadata and
// tensor directory only. Tensor weights are never downloaded or stored.
// ---------------------------------------------------------------------------

enum class tensor_role {
    unknown,
    embedding,
    output_norm,
    lm_head,
    norm,
    layer,
    other,
};

std::string tensor_role_to_string(tensor_role role);
tensor_role   tensor_role_from_string(const std::string & s);

struct tensor_descriptor {
    std::string           name;
    std::string           ggml_type;
    std::vector<uint64_t> dims;
    uint64_t              size_bytes = 0;
    uint64_t              offset     = 0;
    int32_t               layer        = -1;
    tensor_role           role         = tensor_role::unknown;

    nlohmann::json to_json() const;
    static tensor_descriptor from_json(const nlohmann::json & j);
};

struct layer_descriptor {
    int32_t                    layer_index = -1;
    uint64_t                   size_bytes  = 0;
    std::vector<std::string>   tensors;

    nlohmann::json to_json() const;
    static layer_descriptor from_json(const nlohmann::json & j);
};

struct model_manifest {
    std::string architecture;
    uint32_t    n_layer  = 0;
    uint32_t    n_ctx    = 0;
    uint32_t    n_vocab  = 0;
    uint32_t    n_embd   = 0;
    uint32_t    gguf_version = 0;

    std::vector<tensor_descriptor> tensors;
    std::vector<layer_descriptor>  layers;
    std::map<std::string, std::string> special_tensors;

    uint64_t tensor_data_offset   = 0;
    uint64_t metadata_bytes_read  = 0;
    std::string source_file;

    bool empty() const;

    nlohmann::json to_json() const;
    static model_manifest from_json(const nlohmann::json & j);
};

struct manifest_build_result {
    bool           success = false;
    model_manifest manifest;
    std::string    error;
    uint64_t       bytes_read = 0;
};

// Parse a local GGUF file (metadata section only; no tensor weights loaded).
model_manifest build_manifest_from_file(const std::string & path);

// Parse a GGUF metadata buffer. Used by unit tests and remote range fetch.
model_manifest build_manifest_from_buffer(const std::vector<uint8_t> & data, std::string & error);

// Build manifest for a registry record: local file if present, otherwise HTTP Range.
manifest_build_result build_manifest_for_record(
        const dist_model_record & record,
        const std::string & models_dir,
        const std::string & fallback_model_path = "");

// Classify tensor name into layer index and semantic role.
void classify_tensor(const std::string & name, int32_t & layer, tensor_role & role);

// Aggregate per-layer sizes from tensor descriptors.
std::vector<layer_descriptor> build_layer_descriptors(const std::vector<tensor_descriptor> & tensors);
