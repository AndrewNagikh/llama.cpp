#include "synchronization_engine.h"

#include "dist_common.h"
#include "executors/http_range/http_range_executor.h"
#include "node_agent/layer_store/layer_gguf_assembler.h"
#include "../runtime_debug/perf_trace.h"
#include "../runtime_debug/perf_gpu_sampler.h"

#include <atomic>
#include <chrono>
#include <map>
#include <random>
#include <thread>

namespace {

static bool verify_operation_stored(const install_operation & op, layer_store & store) {
    const download_operation & dl = op.download;
    if (!dl.blob_id.empty() && !dl.tensor_name.empty()) {
        return store.verify_blob_tensor(dl.blob_id, dl.tensor_name, dl.checksum);
    }
    return store.verify_layer(op.layer_index, dl.checksum);
}

static executor_result execute_delete(const install_operation & op, layer_store & store) {
    executor_result result;
    if (!op.download.blob_id.empty() && !op.download.tensor_name.empty()) {
        if (!store.remove_blob_tensor(op.download.blob_id, op.download.tensor_name)) {
            result.error = "failed to remove blob tensor";
            return result;
        }
        result.success = true;
        return result;
    }
    if (!store.remove_layer(op.layer_index)) {
        result.error = "failed to remove layer";
        return result;
    }
    result.success = true;
    return result;
}

static executor_result execute_verify(const install_operation & op, layer_store & store) {
    executor_result result;
    if (!verify_operation_stored(op, store)) {
        result.error = "layer verification failed";
        return result;
    }
    result.success = true;
    return result;
}

} // namespace

static const char * install_action_sub(const install_action action) {
    switch (action) {
        case install_action::download: return "download";
        case install_action::verify:   return "verify";
        case install_action::repair:   return "repair";
        case install_action::delete_op: return "delete";
    }
    return "unknown";
}

static std::string install_blob_id(const install_operation & op) {
    if (!op.download.blob_id.empty()) {
        return op.download.blob_id;
    }
    if (op.layer_index >= 0) {
        return "layer:" + std::to_string(op.layer_index);
    }
    return "unknown";
}

static void emit_install_operation_event(
        const install_operation & op,
        const executor_result & result,
        const int64_t dur_us) {
    if (!perf_trace_enabled()) {
        return;
    }
    perf_trace_refresh_context();
    const char * sub = result.success
            ? install_action_sub(op.action)
            : "failed";
    const uint64_t bytes = (op.action == install_action::download ||
                            op.action == install_action::repair)
            ? op.download.tensor_length
            : 0;
    perf_emit_install_span(
            "INSTALL_BLOB",
            sub,
            install_blob_id(op).c_str(),
            op.node_id.c_str(),
            bytes,
            dur_us);
}

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

    std::atomic<bool> any_failed{false};
    std::atomic<size_t> next_index{0};
    std::mutex store_mu;

    const int worker_count = std::max(1, std::min(
            dist_sync_parallelism(),
            static_cast<int>(operations.size())));

    auto run_one = [&](const size_t i) {
        const install_operation & op = operations[i];
        const int64_t t0_us = static_cast<int64_t>(perf_now_us());

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
            std::lock_guard<std::mutex> lock(store_mu);
            result = execute_delete(op, store);
        } else if (op.action == install_action::verify) {
            {
                std::lock_guard<std::mutex> lock(mu_);
                auto it = jobs_.find(job_id);
                if (it != jobs_.end() && i < it->second.operations.size()) {
                    it->second.operations[i].state = sync_operation_state::verifying;
                }
            }
            std::lock_guard<std::mutex> lock(store_mu);
            result = execute_verify(op, store);
        } else if (op.action == install_action::download || op.action == install_action::repair) {
            std::vector<uint8_t> body;
            std::string fetch_error;
            if (!http_range_download_executor::fetch_body(op.download, body, fetch_error)) {
                result.success = false;
                result.error   = fetch_error;
            } else {
                {
                    std::lock_guard<std::mutex> lock(mu_);
                    auto it = jobs_.find(job_id);
                    if (it != jobs_.end() && i < it->second.operations.size()) {
                        it->second.operations[i].state = sync_operation_state::verifying;
                    }
                }
                std::lock_guard<std::mutex> lock(store_mu);
                result = http_range_download_executor::store_body(op, store, std::move(body));
            }
        } else {
            result.error = "unsupported install action";
        }

        const int64_t dur_us = static_cast<int64_t>(perf_now_us()) - t0_us;
        emit_install_operation_event(op, result, dur_us);

        {
            std::lock_guard<std::mutex> lock(mu_);
            auto it = jobs_.find(job_id);
            if (it == jobs_.end()) {
                return;
            }
            if (i >= it->second.operations.size()) {
                return;
            }
            if (result.success) {
                it->second.operations[i].state = sync_operation_state::ready;
            } else {
                it->second.operations[i].state = sync_operation_state::failed;
                it->second.operations[i].error = result.error;
                any_failed.store(true);
            }
        }
    };

    auto worker = [&]() {
        for (;;) {
            const size_t i = next_index.fetch_add(1);
            if (i >= operations.size()) {
                return;
            }
            run_one(i);
        }
    };

    std::vector<std::thread> workers;
    workers.reserve(static_cast<size_t>(worker_count));
    for (int w = 0; w < worker_count; ++w) {
        workers.emplace_back(worker);
    }
    for (auto & t : workers) {
        t.join();
    }

    std::lock_guard<std::mutex> lock(mu_);
    auto it = jobs_.find(job_id);
    if (it == jobs_.end()) {
        return;
    }
    if (any_failed.load()) {
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
    if (perf_trace_has_active_context()) {
        perf_gpu_poll_stop();
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
