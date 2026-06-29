#pragma once

#include "architecture/semantic_blob.h"
#include "architecture/semantic_runtime_descriptor.h"
#include "architecture/worker_requirement.h"

#include <map>
#include <string>

// Per-blob stage requirements derived from worker_descriptor + deploy target.

struct blob_stage_requirements {
    bool required_for_entry  = false;
    bool required_for_middle = false;
    bool required_for_final  = false;
};

std::map<std::string, blob_stage_requirements> compute_blob_stage_requirements(
        const semantic_runtime_descriptor & rt);

bool blob_required_for_role(
        const blob_stage_requirements & req,
        worker_role role);
