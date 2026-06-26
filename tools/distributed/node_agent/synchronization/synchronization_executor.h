#pragma once

#include "orchestrator/install_planner/install_planner.h"
#include "node_agent/layer_store/layer_store.h"

#include <string>

struct executor_result {
    bool        success = false;
    std::string error;
};

// Executor backend interface (Task 9.7).
class synchronization_executor {
public:
    virtual ~synchronization_executor() = default;

    virtual executor_result execute(
            const install_operation & operation,
            layer_store & store) = 0;
};
