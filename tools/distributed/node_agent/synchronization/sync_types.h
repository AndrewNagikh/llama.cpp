#pragma once

#include "orchestrator/install_planner/install_planner.h"

#include "nlohmann/json.hpp"

#include <cstdint>
#include <string>
#include <vector>

// Progress states for a single install operation (Task 9.7).

enum class sync_operation_state {
    queued,
    running,
    verifying,
    ready,
    failed
};

std::string sync_operation_state_to_string(sync_operation_state state);
sync_operation_state sync_operation_state_from_string(const std::string & s);

enum class sync_job_state {
    queued,
    running,
    completed,
    failed
};

std::string sync_job_state_to_string(sync_job_state state);

struct sync_operation_progress {
    std::string           operation_id;
    install_operation     operation;
    sync_operation_state  state = sync_operation_state::queued;
    std::string           error;

    nlohmann::json to_json() const;
};

struct sync_job {
    std::string job_id;
    std::string model_id;
    sync_job_state state = sync_job_state::queued;
    std::vector<sync_operation_progress> operations;
    std::string error;

    int ready_count() const;
    int failed_count() const;
    int total_count() const;

    nlohmann::json to_json() const;
};
