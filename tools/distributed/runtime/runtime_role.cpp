#include "runtime_role.h"

std::string runtime_role_name(const runtime_role role) {
    switch (role) {
        case runtime_role::tokenizer:      return "tokenizer";
        case runtime_role::embedding:      return "embedding";
        case runtime_role::pipeline_stage: return "pipeline_stage";
        case runtime_role::output_head:    return "output_head";
        case runtime_role::sampler:        return "sampler";
        default:                           return "unassigned";
    }
}

runtime_role runtime_role_from_string(const std::string & name) {
    if (name == "tokenizer")      { return runtime_role::tokenizer; }
    if (name == "embedding")      { return runtime_role::embedding; }
    if (name == "pipeline_stage") { return runtime_role::pipeline_stage; }
    if (name == "output_head")    { return runtime_role::output_head; }
    if (name == "sampler")        { return runtime_role::sampler; }
    return runtime_role::unassigned;
}

bool runtime_role_is_service(const runtime_role role) {
    return role == runtime_role::tokenizer ||
           role == runtime_role::embedding ||
           role == runtime_role::output_head ||
           role == runtime_role::sampler;
}

bool runtime_role_is_pipeline(const runtime_role role) {
    return role == runtime_role::pipeline_stage;
}

nlohmann::json runtime_role_to_json(const runtime_role role) {
    return runtime_role_name(role);
}
