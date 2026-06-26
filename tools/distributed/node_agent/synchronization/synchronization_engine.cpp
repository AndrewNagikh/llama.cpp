#include "synchronization_engine.h"

#include "executors/http_range/http_range_executor.h"
#include "node_agent/layer_store/layer_gguf_assembler.h"

#include <chrono>
#include <map>
#include <random>
#include <thread>

namespace {

static executor_result execute_delete(const install_operation & op, layer_store & store) {
    executor_result result;
    if (!store.remove_layer(op.layer_index)) {
        result.error = "failed to remove layer";
        return result;
    }
    result.success = true;
    return result;
}

static executor_result execute_verify(const install_operation & op, layer_store & store) {
    executor_result result;
    const std::string checksum = op.download.checksum;
    if (!store.verify_layer(op.layer_index, checksum)) {
        result.error = "layer verification failed";
        return result;
    }
    result.success = true;
    return result;
}

} // namespace

synchronization_engine::synchronization_engine()
        : executor_(std::make_shared<http_range_download_executor>()) {}

void synchronization_engine::set_executor(std::shared_ptr<synchronization_executor> executor) {
    if (executor) {
        executor_ = std::move(executor);
    }
}

std::string synchronization_engine::make_job_id() {
    static std::mt19937_64 rng{std::random_device{}()};
    const uint64_t n = rng();
    return "sync-" + std::to_string(n);
}

std::string synchronization_engine::start_job(
        const std::string & model_id,
        const std::vector<install_operation> & operations,
        layer_store & store,
        const model_manifest * manifest) {
    const std::string job_id = make_job_id();

    sync_job job;
    job.job_id  = job_id;
    job.model_id = model_id;
    job.state   = sync_job_state::queued;

    int idx = 0;
    for (const auto & op : operations) {
        sync_operation_progress progress;
        progress.operation_id = job_id + "-op-" + std::to_string(idx++);
        progress.operation    = op;
        progress.state        = sync_operation_state::queued;
        job.operations.push_back(std::move(progress));
    }

    {
        std::lock_guard<std::mutex> lock(mu_);
        jobs_.emplace(job_id, job);
    }

    model_manifest manifest_copy;
    bool has_manifest = false;
    if (manifest != nullptr) {
        manifest_copy = *manifest;
        has_manifest  = true;
        store.save_manifest(*manifest);
    }

    std::thread(&synchronization_engine::run_job,
            this,
            job_id,
            operations,
            store,
            manifest_copy,
            has_manifest).detach();

    return job_id;
}

void synchronization_engine::run_job(
        const std::string job_id,
        std::vector<install_operation> operations,
        layer_store store,
        const model_manifest manifest,
        const bool has_manifest) {
  (void) operations;
    if (has_manifest) {
        store.save_manifest(manifest);
    }

    {
        std::lock_guard<std::mutex> lock(mu_);
        auto it = jobs_.find(job_id);
        if (it == jobs_.end()) {
            return;
        }
        it->second.state = sync_job_state::running;
    }

    bool any_failed = false;

    for (size_t i = 0; i < operations.size(); ++i) {
        const install_operation & op = operations[i];

        {
            std::lock_guard<std::mutex> lock(mu_);
            auto it = jobs_.find(job_id);
            if (it == jobs_.end()) {
                return;
            }
            if (i < it->second.operations.size()) {
                it->second.operations[i].state = sync_operation_state::running;
            }
        }

        executor_result result;
        if (op.action == install_action::delete_op) {
            result = execute_delete(op, store);
        } else if (op.action == install_action::verify) {
            {
                std::lock_guard<std::mutex> lock(mu_);
                auto it = jobs_.find(job_id);
                if (it != jobs_.end() && i < it->second.operations.size()) {
                    it->second.operations[i].state = sync_operation_state::verifying;
                }
            }
            result = execute_verify(op, store);
        } else if (op.action == install_action::download || op.action == install_action::repair) {
            result = executor_->execute(op, store);
            if (result.success) {
                std::lock_guard<std::mutex> lock(mu_);
                auto it = jobs_.find(job_id);
                if (it != jobs_.end() && i < it->second.operations.size()) {
                    it->second.operations[i].state = sync_operation_state::verifying;
                }
                if (!store.verify_layer(op.layer_index, op.download.checksum)) {
                    result.success = false;
                    result.error   = "verification failed after download";
                }
            }
        } else {
            result.error = "unsupported install action";
        }

        {
            std::lock_guard<std::mutex> lock(mu_);
            auto it = jobs_.find(job_id);
            if (it == jobs_.end()) {
                return;
            }
            if (i >= it->second.operations.size()) {
                continue;
            }
            if (result.success) {
                it->second.operations[i].state = sync_operation_state::ready;
            } else {
                it->second.operations[i].state = sync_operation_state::failed;
                it->second.operations[i].error = result.error;
                any_failed = true;
            }
        }
    }

    std::lock_guard<std::mutex> lock(mu_);
    auto it = jobs_.find(job_id);
    if (it == jobs_.end()) {
        return;
    }
    if (any_failed) {
        it->second.state = sync_job_state::failed;
        it->second.error = "one or more operations failed";
    } else {
        it->second.state = sync_job_state::completed;
        if (has_manifest) {
            std::string source_url;
            for (const auto & op : operations) {
                if (!op.download.source_url.empty()) {
                    source_url = op.download.source_url;
                    break;
                }
            }
            if (!source_url.empty()) {
                layer_store_cache_metadata(store, manifest, source_url);
            }
        }
    }
}

std::optional<sync_job> synchronization_engine::get_job(const std::string & job_id) const {
    std::lock_guard<std::mutex> lock(mu_);
    const auto it = jobs_.find(job_id);
    if (it == jobs_.end()) {
        return std::nullopt;
    }
    return it->second;
}

bool synchronization_engine::wait_job(const std::string & job_id, const int timeout_ms) const {
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(timeout_ms);
    while (std::chrono::steady_clock::now() < deadline) {
        const auto job = get_job(job_id);
        if (!job.has_value()) {
            return false;
        }
        if (job->state == sync_job_state::completed) {
            return true;
        }
        if (job->state == sync_job_state::failed) {
            return false;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }
    return false;
}
