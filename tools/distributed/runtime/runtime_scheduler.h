#pragma once

#include "runtime_descriptor.h"
#include "runtime_graph.h"

#include <string>
#include <vector>

enum class runtime_scheduler_operation {
    start_prefill,
    commit_prefill,
    start_decode_step,
    commit_decode_step,
};

struct runtime_scheduler_validation {
    bool valid = false;
    std::vector<std::string> errors;

    bool ok() const { return valid && errors.empty(); }
};

runtime_scheduler_validation validate_runtime_scheduler_operation(
        const runtime_descriptor & desc,
        const runtime_graph & graph,
        runtime_scheduler_operation op);

runtime_scheduler_validation validate_runtime_scheduler_prefill_decode(
        const runtime_descriptor & desc,
        const runtime_graph & graph);
