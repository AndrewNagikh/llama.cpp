#pragma once

#include "runtime_graph.h"
#include "runtime_role.h"

#include "nlohmann/json.hpp"

#include <cstdint>
#include <string>
#include <vector>

enum class runtime_resource_kind : uint32_t {
    unknown             = 0,
    tokenizer_model     = 1,
    tokenizer_vocab      = 2,
    special_token_map    = 3,
    token_embedding      = 4,
    position_embedding   = 5,
    rope_parameters      = 6,
    input_norm           = 7,
    transformer_block    = 8,
    output_norm          = 9,
    lm_head              = 10,
    sampling_metadata    = 11,
    vocabulary_metadata  = 12,
    kv_cache_layout      = 13,
};

std::string runtime_resource_kind_name(runtime_resource_kind kind);
runtime_resource_kind runtime_resource_kind_from_string(const std::string & name);

struct runtime_execution_capabilities {
    bool can_move        = false;
    bool can_migrate     = false;
    bool can_restart     = false;
    bool can_replicate   = false;
    bool can_parallelize = false;
    bool can_colocate    = false;
    bool can_cache       = false;
    bool can_share       = false;
    bool can_stream      = false;
    bool can_pipeline    = false;
    bool can_prefetch    = false;
    bool can_resume      = false;
    bool can_checkpoint  = false;

    nlohmann::json to_json() const;
};

struct runtime_cost_descriptor {
    uint64_t memory_static_bytes            = 0;
    uint64_t memory_runtime_bytes_per_token = 0;
    uint64_t compute_weight_prefill         = 0;
    uint64_t compute_weight_decode          = 0;
    uint64_t network_bytes_per_token        = 0;

    nlohmann::json to_json() const;
};

struct runtime_service_descriptor {
    std::string name;
    runtime_role role = runtime_role::unassigned;

    std::vector<runtime_role> dependencies;
    std::vector<std::string> required_resources;
    std::vector<std::string> optional_resources;

    std::string input_type;
    std::string output_type;
    std::string failure_policy;

    runtime_cost_descriptor cost;
    runtime_execution_capabilities capabilities;

    bool supports_parallel    = false;
    bool supports_replication = false;
    bool supports_colocation  = false;

    nlohmann::json to_json() const;
};

struct runtime_semantic_resource_descriptor {
    std::string id;
    runtime_resource_kind kind = runtime_resource_kind::unknown;

    bool required   = true;
    bool shareable  = false;
    bool shardable  = false;
    bool replicated = false;

    uint64_t size_bytes         = 0;
    uint64_t runtime_size_bytes = 0;

    std::string dtype;
    std::string layout;
    std::string tied_to;

    nlohmann::json to_json() const;
};

struct runtime_dependency_descriptor {
    runtime_role from = runtime_role::unassigned;
    runtime_role to   = runtime_role::unassigned;

    std::string edge_kind;
    std::string data_contract;

    bool ordering_required = true;
    bool streaming_supported = false;
    std::string failure_policy;

    nlohmann::json to_json() const;
};

struct runtime_descriptor {
    std::string descriptor_id;
    std::string schema_version = "1.0";
    std::string model_id;
    std::string architecture_descriptor_id;
    std::string created_from_manifest_id;
    std::string compatibility_version = "1.0";

    std::vector<runtime_service_descriptor> services;
    std::vector<runtime_semantic_resource_descriptor> resources;
    std::vector<runtime_dependency_descriptor> dependencies;

    nlohmann::json to_json() const;
};

struct runtime_descriptor_validation {
    bool valid = false;
    std::vector<std::string> errors;

    bool ok() const { return valid && errors.empty(); }
};

struct runtime_execution_graph_validation {
    bool valid = false;
    std::vector<std::string> errors;

    bool ok() const { return valid && errors.empty(); }
};

const runtime_service_descriptor * find_runtime_service(
        const runtime_descriptor & desc,
        runtime_role role);

const runtime_semantic_resource_descriptor * find_runtime_resource(
        const runtime_descriptor & desc,
        const std::string & id);

runtime_descriptor_validation validate_runtime_descriptor(
        const runtime_descriptor & desc);

runtime_execution_graph_validation validate_runtime_graph_against_descriptor(
        const runtime_descriptor & desc,
        const runtime_graph & graph);

runtime_descriptor make_basic_text_generation_runtime_descriptor(
        const std::string & model_id,
        const std::string & architecture_descriptor_id,
        int32_t n_layers);
