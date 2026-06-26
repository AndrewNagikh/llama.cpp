#pragma once

#include "layer_store/layer_store.h"
#include "manifest_builder/manifest_builder.h"
#include "verification_types.h"

#include <string>
#include <vector>

enum class worker_verify_role {
    entry,
    middle,
    final,
    full,
};

std::string worker_verify_role_to_string(worker_verify_role role);

struct worker_tensor_plan {
    worker_verify_role role       = worker_verify_role::full;
    int32_t            layer_start = 0;
    int32_t            layer_end   = 0;
    bool               include_embedding = false;
    bool               include_output    = false;
};

inline worker_tensor_plan make_worker_tensor_plan(
        const worker_verify_role role,
        const int32_t layer_start,
        const int32_t layer_end) {
    worker_tensor_plan plan;
    plan.role         = role;
    plan.layer_start  = layer_start;
    plan.layer_end    = layer_end;
    switch (role) {
        case worker_verify_role::entry:
            plan.include_embedding = true;
            plan.include_output    = false;
            break;
        case worker_verify_role::middle:
            plan.include_embedding = false;
            plan.include_output    = false;
            break;
        case worker_verify_role::final:
            plan.include_embedding = false;
            plan.include_output    = true;
            break;
        case worker_verify_role::full:
            plan.include_embedding = true;
            plan.include_output    = true;
            break;
    }
    return plan;
}

bool gguf_tensor_included(
        const tensor_descriptor & t,
        int32_t layer_start,
        int32_t layer_end,
        bool include_embedding,
        bool include_output,
        const model_manifest * manifest = nullptr);

std::vector<std::string> list_tensors_for_worker(
        const model_manifest & manifest,
        const worker_tensor_plan & plan);

struct materialized_special_flags {
    bool embedding   = false;
    bool output      = false;
    bool output_norm = false;
};

materialized_special_flags inspect_materialized_special_tensors(
        const std::string & gguf_path,
        const model_manifest & manifest = {});

verify_check_result verify_worker_tensor_plan(
        const model_manifest & manifest,
        const worker_tensor_plan & plan);

verify_check_result verify_materialized_structure_for_role(
        const std::string & gguf_path,
        const worker_tensor_plan & plan,
        const model_manifest & manifest);

verify_check_result verify_worker_store_assignment(
        const layer_store & store,
        const model_manifest & manifest,
        const worker_tensor_plan & plan);
