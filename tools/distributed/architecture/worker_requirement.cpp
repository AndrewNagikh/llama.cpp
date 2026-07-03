#include "worker_requirement.h"

#include "runtime/runtime_role.h"

std::string worker_role_to_string(const worker_role role) {
    switch (role) {
        case worker_role::tokenizer:       return "TOKENIZER";
        case worker_role::embedding:       return "EMBEDDING";
        case worker_role::pipeline_stage:  return "PIPELINE_STAGE";
        case worker_role::output_head:     return "OUTPUT_HEAD";
        case worker_role::sampler:         return "SAMPLER";
        case worker_role::entry:           return "ENTRY";
        case worker_role::middle:          return "MIDDLE";
        case worker_role::final:           return "FINAL";
        case worker_role::full:            return "FULL";
    }
    return "UNKNOWN";
}

worker_materialize_plan materialize_plan_for_worker(
        const std::vector<worker_requirement> & requirements,
        const worker_role role,
        const int32_t layer_start,
        const int32_t layer_end) {
    worker_materialize_plan plan;
    plan.role         = role;
    plan.layer_start  = layer_start;
    plan.layer_end    = layer_end;

    for (const auto & req : requirements) {
        if (req.role != role) {
            continue;
        }
        plan.required_blobs.insert(
                plan.required_blobs.end(),
                req.required_blobs.begin(),
                req.required_blobs.end());
    }
    return plan;
}

worker_role worker_role_from_runtime_role(const runtime_role role) {
    switch (role) {
        case runtime_role::tokenizer:      return worker_role::tokenizer;
        case runtime_role::embedding:      return worker_role::embedding;
        case runtime_role::pipeline_stage: return worker_role::pipeline_stage;
        case runtime_role::output_head:    return worker_role::output_head;
        case runtime_role::sampler:        return worker_role::sampler;
        default:                           return worker_role::pipeline_stage;
    }
}
