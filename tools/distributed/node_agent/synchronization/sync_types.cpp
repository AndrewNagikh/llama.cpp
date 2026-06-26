#include "sync_types.h"

using json = nlohmann::json;

std::string sync_operation_state_to_string(const sync_operation_state state) {
    switch (state) {
        case sync_operation_state::queued:    return "QUEUED";
        case sync_operation_state::running:   return "RUNNING";
        case sync_operation_state::verifying: return "VERIFYING";
        case sync_operation_state::ready:     return "READY";
        case sync_operation_state::failed:    return "FAILED";
    }
    return "QUEUED";
}

sync_operation_state sync_operation_state_from_string(const std::string & s) {
    if (s == "RUNNING")   return sync_operation_state::running;
    if (s == "VERIFYING") return sync_operation_state::verifying;
    if (s == "READY")     return sync_operation_state::ready;
    if (s == "FAILED")    return sync_operation_state::failed;
    return sync_operation_state::queued;
}

std::string sync_job_state_to_string(const sync_job_state state) {
    switch (state) {
        case sync_job_state::queued:    return "QUEUED";
        case sync_job_state::running:   return "RUNNING";
        case sync_job_state::completed: return "COMPLETED";
        case sync_job_state::failed:    return "FAILED";
    }
    return "QUEUED";
}

json sync_operation_progress::to_json() const {
    json j = {
        { "operation_id", operation_id },
        { "state", sync_operation_state_to_string(state) },
        { "action", install_action_to_string(operation.action) },
        { "node_id", operation.node_id },
        { "layer_index", operation.layer_index },
    };
    if (!error.empty()) {
        j["error"] = error;
    }
    if (operation.action == install_action::download ||
            operation.action == install_action::repair) {
        j["download"] = operation.download.to_json();
    }
    return j;
}

int sync_job::ready_count() const {
    int n = 0;
    for (const auto & op : operations) {
        if (op.state == sync_operation_state::ready) {
            ++n;
        }
    }
    return n;
}

int sync_job::failed_count() const {
    int n = 0;
    for (const auto & op : operations) {
        if (op.state == sync_operation_state::failed) {
            ++n;
        }
    }
    return n;
}

int sync_job::total_count() const {
    return static_cast<int>(operations.size());
}

json sync_job::to_json() const {
    json ops = json::array();
    for (const auto & op : operations) {
        ops.push_back(op.to_json());
    }
    json j = {
        { "job_id", job_id },
        { "model_id", model_id },
        { "state", sync_job_state_to_string(state) },
        { "ready_count", ready_count() },
        { "failed_count", failed_count() },
        { "total_count", total_count() },
        { "operations", ops },
    };
    if (!error.empty()) {
        j["error"] = error;
    }
    return j;
}
