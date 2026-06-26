#include "worker_requirement.h"

std::string worker_role_to_string(const worker_role role) {
    switch (role) {
        case worker_role::entry:  return "ENTRY";
        case worker_role::middle: return "MIDDLE";
        case worker_role::final:  return "FINAL";
        case worker_role::full:   return "FULL";
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
