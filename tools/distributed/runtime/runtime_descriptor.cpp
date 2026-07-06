#include "runtime_descriptor.h"

#include <algorithm>
#include <map>
#include <set>

std::string runtime_resource_kind_name(const runtime_resource_kind kind) {
    switch (kind) {
        case runtime_resource_kind::tokenizer_model:    return "tokenizer_model";
        case runtime_resource_kind::tokenizer_vocab:     return "tokenizer_vocab";
        case runtime_resource_kind::special_token_map:   return "special_token_map";
        case runtime_resource_kind::token_embedding:     return "token_embedding";
        case runtime_resource_kind::position_embedding:  return "position_embedding";
        case runtime_resource_kind::rope_parameters:     return "rope_parameters";
        case runtime_resource_kind::input_norm:          return "input_norm";
        case runtime_resource_kind::transformer_block:   return "transformer_block";
        case runtime_resource_kind::output_norm:         return "output_norm";
        case runtime_resource_kind::lm_head:             return "lm_head";
        case runtime_resource_kind::sampling_metadata:   return "sampling_metadata";
        case runtime_resource_kind::vocabulary_metadata: return "vocabulary_metadata";
        case runtime_resource_kind::kv_cache_layout:     return "kv_cache_layout";
        case runtime_resource_kind::unknown:
        default:                                        return "unknown";
    }
}

runtime_resource_kind runtime_resource_kind_from_string(const std::string & name) {
    if (name == "tokenizer_model")    { return runtime_resource_kind::tokenizer_model; }
    if (name == "tokenizer_vocab")     { return runtime_resource_kind::tokenizer_vocab; }
    if (name == "special_token_map")   { return runtime_resource_kind::special_token_map; }
    if (name == "token_embedding")     { return runtime_resource_kind::token_embedding; }
    if (name == "position_embedding")  { return runtime_resource_kind::position_embedding; }
    if (name == "rope_parameters")     { return runtime_resource_kind::rope_parameters; }
    if (name == "input_norm")          { return runtime_resource_kind::input_norm; }
    if (name == "transformer_block")   { return runtime_resource_kind::transformer_block; }
    if (name == "output_norm")         { return runtime_resource_kind::output_norm; }
    if (name == "lm_head")             { return runtime_resource_kind::lm_head; }
    if (name == "sampling_metadata")   { return runtime_resource_kind::sampling_metadata; }
    if (name == "vocabulary_metadata") { return runtime_resource_kind::vocabulary_metadata; }
    if (name == "kv_cache_layout")     { return runtime_resource_kind::kv_cache_layout; }
    return runtime_resource_kind::unknown;
}

nlohmann::json runtime_execution_capabilities::to_json() const {
    return {
        { "can_move", can_move },
        { "can_migrate", can_migrate },
        { "can_restart", can_restart },
        { "can_replicate", can_replicate },
        { "can_parallelize", can_parallelize },
        { "can_colocate", can_colocate },
        { "can_cache", can_cache },
        { "can_share", can_share },
        { "can_stream", can_stream },
        { "can_pipeline", can_pipeline },
        { "can_prefetch", can_prefetch },
        { "can_resume", can_resume },
        { "can_checkpoint", can_checkpoint },
    };
}

nlohmann::json runtime_cost_descriptor::to_json() const {
    return {
        { "memory_static_bytes", memory_static_bytes },
        { "memory_runtime_bytes_per_token", memory_runtime_bytes_per_token },
        { "compute_weight_prefill", compute_weight_prefill },
        { "compute_weight_decode", compute_weight_decode },
        { "network_bytes_per_token", network_bytes_per_token },
    };
}

nlohmann::json runtime_service_descriptor::to_json() const {
    std::vector<std::string> deps;
    deps.reserve(dependencies.size());
    for (const runtime_role dep : dependencies) {
        deps.push_back(runtime_role_name(dep));
    }

    return {
        { "name", name },
        { "role", runtime_role_name(role) },
        { "dependencies", deps },
        { "required_resources", required_resources },
        { "optional_resources", optional_resources },
        { "input_type", input_type },
        { "output_type", output_type },
        { "failure_policy", failure_policy },
        { "cost", cost.to_json() },
        { "capabilities", capabilities.to_json() },
        { "supports_parallel", supports_parallel },
        { "supports_replication", supports_replication },
        { "supports_colocation", supports_colocation },
    };
}

nlohmann::json runtime_semantic_resource_descriptor::to_json() const {
    return {
        { "id", id },
        { "kind", runtime_resource_kind_name(kind) },
        { "required", required },
        { "shareable", shareable },
        { "shardable", shardable },
        { "replicated", replicated },
        { "size_bytes", size_bytes },
        { "runtime_size_bytes", runtime_size_bytes },
        { "dtype", dtype },
        { "layout", layout },
        { "tied_to", tied_to },
    };
}

nlohmann::json runtime_dependency_descriptor::to_json() const {
    return {
        { "from", runtime_role_name(from) },
        { "to", runtime_role_name(to) },
        { "edge_kind", edge_kind },
        { "data_contract", data_contract },
        { "ordering_required", ordering_required },
        { "streaming_supported", streaming_supported },
        { "failure_policy", failure_policy },
    };
}

nlohmann::json runtime_descriptor::to_json() const {
    nlohmann::json service_json = nlohmann::json::array();
    for (const runtime_service_descriptor & service : services) {
        service_json.push_back(service.to_json());
    }

    nlohmann::json resource_json = nlohmann::json::array();
    for (const runtime_semantic_resource_descriptor & resource : resources) {
        resource_json.push_back(resource.to_json());
    }

    nlohmann::json dependency_json = nlohmann::json::array();
    for (const runtime_dependency_descriptor & dependency : dependencies) {
        dependency_json.push_back(dependency.to_json());
    }

    return {
        { "descriptor_id", descriptor_id },
        { "schema_version", schema_version },
        { "model_id", model_id },
        { "architecture_descriptor_id", architecture_descriptor_id },
        { "created_from_manifest_id", created_from_manifest_id },
        { "compatibility_version", compatibility_version },
        { "services", service_json },
        { "resources", resource_json },
        { "dependencies", dependency_json },
    };
}

const runtime_service_descriptor * find_runtime_service(
        const runtime_descriptor & desc,
        const runtime_role role) {
    for (const runtime_service_descriptor & service : desc.services) {
        if (service.role == role) {
            return &service;
        }
    }
    return nullptr;
}

const runtime_semantic_resource_descriptor * find_runtime_resource(
        const runtime_descriptor & desc,
        const std::string & id) {
    for (const runtime_semantic_resource_descriptor & resource : desc.resources) {
        if (resource.id == id) {
            return &resource;
        }
    }
    return nullptr;
}

static void add_error(
        runtime_descriptor_validation & out,
        const std::string & message) {
    out.errors.push_back(message);
}

runtime_descriptor_validation validate_runtime_descriptor(
        const runtime_descriptor & desc) {
    runtime_descriptor_validation out{};

    if (desc.schema_version.empty()) {
        add_error(out, "schema_version is required");
    } else if (desc.schema_version != "1.0") {
        add_error(out, "unsupported schema_version: " + desc.schema_version);
    }
    if (desc.descriptor_id.empty()) {
        add_error(out, "descriptor_id is required");
    }
    if (desc.model_id.empty()) {
        add_error(out, "model_id is required");
    }
    if (desc.architecture_descriptor_id.empty()) {
        add_error(out, "architecture_descriptor_id is required");
    }

    std::set<runtime_role> service_roles;
    for (const runtime_service_descriptor & service : desc.services) {
        if (service.role == runtime_role::unassigned) {
            add_error(out, "service has unassigned role: " + service.name);
        }
        if (service.name.empty()) {
            add_error(out, "service name is required");
        }
        if (service.input_type.empty() && service.role != runtime_role::tokenizer) {
            add_error(out, "service input_type is required: " + service.name);
        }
        if (service.output_type.empty() && service.role != runtime_role::sampler) {
            add_error(out, "service output_type is required: " + service.name);
        }
        if (service.failure_policy.empty()) {
            add_error(out, "service failure_policy is required: " + service.name);
        }
        if (!service_roles.insert(service.role).second) {
            add_error(out, "duplicate service role: " + runtime_role_name(service.role));
        }
    }

    const runtime_role required_roles[] = {
        runtime_role::tokenizer,
        runtime_role::embedding,
        runtime_role::pipeline_stage,
        runtime_role::output_head,
        runtime_role::sampler,
    };
    for (const runtime_role role : required_roles) {
        if (service_roles.find(role) == service_roles.end()) {
            add_error(out, "missing required service: " + runtime_role_name(role));
        }
    }

    std::set<std::string> resource_ids;
    for (const runtime_semantic_resource_descriptor & resource : desc.resources) {
        if (resource.id.empty()) {
            add_error(out, "resource id is required");
        }
        if (resource.kind == runtime_resource_kind::unknown) {
            add_error(out, "resource kind is unknown: " + resource.id);
        }
        if (!resource_ids.insert(resource.id).second) {
            add_error(out, "duplicate resource id: " + resource.id);
        }
        if (!resource.tied_to.empty() && find_runtime_resource(desc, resource.tied_to) == nullptr) {
            add_error(out, "resource tied_to target missing: " + resource.id);
        }
    }

    for (const runtime_service_descriptor & service : desc.services) {
        for (const runtime_role dep : service.dependencies) {
            if (service_roles.find(dep) == service_roles.end()) {
                add_error(out,
                        "service dependency missing: " +
                        service.name +
                        " -> " +
                        runtime_role_name(dep));
            }
        }
        for (const std::string & resource_id : service.required_resources) {
            if (resource_ids.find(resource_id) == resource_ids.end()) {
                add_error(out,
                        "required resource missing: " +
                        service.name +
                        " -> " +
                        resource_id);
            }
        }
    }

    for (const runtime_dependency_descriptor & dependency : desc.dependencies) {
        const runtime_service_descriptor * from_service = find_runtime_service(desc, dependency.from);
        const runtime_service_descriptor * to_service = find_runtime_service(desc, dependency.to);
        if (from_service == nullptr) {
            add_error(out, "dependency from service missing: " + runtime_role_name(dependency.from));
        }
        if (to_service == nullptr) {
            add_error(out, "dependency to service missing: " + runtime_role_name(dependency.to));
        }
        if (dependency.edge_kind.empty()) {
            add_error(out, "dependency edge_kind is required");
        }
        if (dependency.data_contract.empty()) {
            add_error(out, "dependency data_contract is required");
        }
        if (dependency.failure_policy.empty()) {
            add_error(out,
                    "dependency failure_policy is required: " +
                    runtime_role_name(dependency.from) +
                    " -> " +
                    runtime_role_name(dependency.to));
        }
        if (from_service != nullptr && to_service != nullptr) {
            if (!from_service->output_type.empty() &&
                    dependency.data_contract != from_service->output_type) {
                add_error(out,
                        "dependency data_contract does not match producer output: " +
                        runtime_role_name(dependency.from) +
                        " -> " +
                        runtime_role_name(dependency.to));
            }
            if (!to_service->input_type.empty() &&
                    dependency.data_contract != to_service->input_type) {
                add_error(out,
                        "dependency data_contract does not match consumer input: " +
                        runtime_role_name(dependency.from) +
                        " -> " +
                        runtime_role_name(dependency.to));
            }
        }
    }

    out.valid = out.errors.empty();
    return out;
}

static void add_graph_error(
        runtime_execution_graph_validation & out,
        const std::string & message) {
    out.errors.push_back(message);
}

runtime_execution_graph_validation validate_runtime_graph_against_descriptor(
        const runtime_descriptor & desc,
        const runtime_graph & graph) {
    runtime_execution_graph_validation out{};

    const runtime_descriptor_validation desc_validation =
            validate_runtime_descriptor(desc);
    if (!desc_validation.ok()) {
        for (const std::string & error : desc_validation.errors) {
            add_graph_error(out, "descriptor invalid: " + error);
        }
        out.valid = false;
        return out;
    }

    if (graph.model_id.empty()) {
        add_graph_error(out, "graph model_id is required");
    }
    if (graph.assignments.empty()) {
        add_graph_error(out, "graph assignments are required");
    }

    std::set<runtime_role> assigned_roles;
    std::map<int32_t, int32_t> pipeline_blocks;
    int32_t pipeline_stage_count = 0;
    for (const runtime_role_assignment & assignment : graph.assignments) {
        if (assignment.role == runtime_role::unassigned) {
            add_graph_error(out, "assignment has unassigned role");
        }
        if (assignment.node_id.empty()) {
            add_graph_error(out, "assignment node_id is required");
        }
        assigned_roles.insert(assignment.role);
        if (assignment.role == runtime_role::pipeline_stage) {
            ++pipeline_stage_count;
            if (assignment.layer_end <= assignment.layer_start) {
                add_graph_error(out, "pipeline stage has invalid block range");
            }
            for (int32_t layer = assignment.layer_start; layer < assignment.layer_end; ++layer) {
                ++pipeline_blocks[layer];
            }
        }
    }

    for (const runtime_service_descriptor & service : desc.services) {
        if (assigned_roles.find(service.role) == assigned_roles.end()) {
            add_graph_error(out, "missing graph assignment for service: " + service.name);
        }
        for (const runtime_role dependency : service.dependencies) {
            if (assigned_roles.find(dependency) == assigned_roles.end()) {
                add_graph_error(out,
                        "missing graph assignment for dependency: " +
                        service.name +
                        " -> " +
                        runtime_role_name(dependency));
            }
        }
    }

    if (pipeline_stage_count <= 0) {
        add_graph_error(out, "at least one pipeline stage assignment is required");
    }
    if (graph.n_layers <= 0) {
        add_graph_error(out, "graph n_layers must be positive");
    } else {
        for (int32_t layer = 0; layer < graph.n_layers; ++layer) {
            const auto it = pipeline_blocks.find(layer);
            if (it == pipeline_blocks.end()) {
                add_graph_error(out, "pipeline block not covered: " + std::to_string(layer));
            } else if (it->second != 1) {
                add_graph_error(out, "pipeline block covered multiple times: " + std::to_string(layer));
            }
        }
        for (const auto & item : pipeline_blocks) {
            if (item.first < 0 || item.first >= graph.n_layers) {
                add_graph_error(out, "pipeline block outside graph range: " + std::to_string(item.first));
            }
        }
    }

    out.valid = out.errors.empty();
    return out;
}

static runtime_semantic_resource_descriptor resource(
        const std::string & id,
        const runtime_resource_kind kind,
        const bool shareable) {
    runtime_semantic_resource_descriptor r{};
    r.id = id;
    r.kind = kind;
    r.required = true;
    r.shareable = shareable;
    return r;
}

static runtime_service_descriptor service(
        const std::string & name,
        const runtime_role role,
        const std::vector<runtime_role> & dependencies,
        const std::vector<std::string> & required_resources,
        const std::string & input_type,
        const std::string & output_type) {
    runtime_service_descriptor s{};
    s.name = name;
    s.role = role;
    s.dependencies = dependencies;
    s.required_resources = required_resources;
    s.input_type = input_type;
    s.output_type = output_type;
    s.failure_policy = "fail_session";
    s.capabilities.can_move = true;
    s.capabilities.can_restart = true;
    s.capabilities.can_colocate = true;
    s.supports_colocation = true;
    return s;
}

runtime_descriptor make_basic_text_generation_runtime_descriptor(
        const std::string & model_id,
        const std::string & architecture_descriptor_id,
        const int32_t n_layers) {
    runtime_descriptor desc{};
    desc.model_id = model_id;
    desc.architecture_descriptor_id = architecture_descriptor_id;
    desc.created_from_manifest_id = model_id + ":manifest";
    desc.descriptor_id = model_id + ":" + architecture_descriptor_id + ":runtime:1.0";

    desc.resources = {
        resource("tokenizer.model", runtime_resource_kind::tokenizer_model, false),
        resource("tokenizer.vocab", runtime_resource_kind::tokenizer_vocab, true),
        resource("tokenizer.special_tokens", runtime_resource_kind::special_token_map, true),
        resource("embedding.token", runtime_resource_kind::token_embedding, false),
        resource("embedding.rope", runtime_resource_kind::rope_parameters, true),
        resource("embedding.input_norm", runtime_resource_kind::input_norm, false),
        resource("pipeline.blocks", runtime_resource_kind::transformer_block, false),
        resource("pipeline.kv_cache", runtime_resource_kind::kv_cache_layout, true),
        resource("output.norm", runtime_resource_kind::output_norm, false),
        resource("output.lm_head", runtime_resource_kind::lm_head, false),
        resource("sampler.metadata", runtime_resource_kind::sampling_metadata, true),
        resource("vocabulary.metadata", runtime_resource_kind::vocabulary_metadata, true),
    };

    runtime_service_descriptor tokenizer = service(
            "Tokenizer",
            runtime_role::tokenizer,
            {},
            { "tokenizer.model", "tokenizer.vocab", "tokenizer.special_tokens" },
            "",
            "token_ids");
    tokenizer.capabilities.can_replicate = true;
    tokenizer.supports_replication = true;

    runtime_service_descriptor embedding = service(
            "Embedding",
            runtime_role::embedding,
            { runtime_role::tokenizer },
            { "embedding.token", "embedding.rope", "embedding.input_norm" },
            "token_ids",
            "hidden_states");

    runtime_service_descriptor pipeline = service(
            "PipelineStage",
            runtime_role::pipeline_stage,
            { runtime_role::embedding },
            { "pipeline.blocks", "pipeline.kv_cache", "embedding.rope" },
            "hidden_states",
            "hidden_states");
    pipeline.capabilities.can_pipeline = n_layers > 1;
    pipeline.supports_parallel = n_layers > 1;

    runtime_service_descriptor output = service(
            "OutputHead",
            runtime_role::output_head,
            { runtime_role::pipeline_stage },
            { "output.norm", "output.lm_head", "vocabulary.metadata" },
            "hidden_states",
            "logits");

    runtime_service_descriptor sampler = service(
            "Sampler",
            runtime_role::sampler,
            { runtime_role::output_head },
            { "sampler.metadata", "vocabulary.metadata", "tokenizer.special_tokens" },
            "logits",
            "");
    sampler.capabilities.can_replicate = true;
    sampler.supports_replication = true;

    desc.services = { tokenizer, embedding, pipeline, output, sampler };

    desc.dependencies = {
        { runtime_role::tokenizer, runtime_role::embedding, "token_ids", "token_ids", true, false, "retry_or_fail_session" },
        { runtime_role::embedding, runtime_role::pipeline_stage, "hidden_states", "hidden_states", true, true, "retry_or_fail_session" },
        { runtime_role::pipeline_stage, runtime_role::output_head, "hidden_states", "hidden_states", true, true, "retry_or_fail_session" },
        { runtime_role::output_head, runtime_role::sampler, "logits", "logits", true, false, "retry_or_fail_session" },
    };

    return desc;
}
