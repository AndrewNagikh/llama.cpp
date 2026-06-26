#pragma once

#include "sync_types.h"
#include "synchronization_executor.h"
#include "node_agent/layer_store/layer_store.h"
#include "manifest_builder/manifest_builder.h"

#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <vector>

// Executes install operations against the local Layer Store (Task 9.7).

class synchronization_engine {
public:
    synchronization_engine();

    std::string start_job(
            const std::string & model_id,
            const std::vector<install_operation> & operations,
            layer_store & store,
            const model_manifest * manifest = nullptr);

    std::optional<sync_job> get_job(const std::string & job_id) const;
    bool wait_job(const std::string & job_id, int timeout_ms) const;

    void set_executor(std::shared_ptr<synchronization_executor> executor);

private:
    mutable std::mutex mu_;
    std::map<std::string, sync_job> jobs_;
    std::shared_ptr<synchronization_executor> executor_;

    static std::string make_job_id();
    void run_job(
            std::string job_id,
            std::vector<install_operation> operations,
            layer_store store,
            const model_manifest manifest,
            bool has_manifest);
};
