#include "orchestrator/model_registry.h"

#include <cassert>
#include <iostream>

int main() {
    cluster_model_registry registry;

    dist_model_record record{};
    record.model_id   = "llama-3.2-1b";
    record.source     = "huggingface";
    record.repository = "hugging-quants/Llama-3.2-1B-Instruct-Q4_K_M-GGUF";
    record.revision   = "main";
    record.status     = dist_model_status::discovered;
    registry.add_or_update(record);

    provider_discovery_result result{};
    result.success    = true;
    result.provider   = "huggingface";
    result.repository = record.repository;
    result.revision   = "main";

    remote_model_file file{};
    file.filename     = "llama-3.2-1b-instruct-q4_k_m.gguf";
    file.size_bytes   = 807690656;
    file.etag         = "etag";
    file.sha256       = "sha256";
    file.download_url = "https://huggingface.co/.../resolve/main/...";
    result.files.push_back(file);

    dist_model_record updated{};
    [[maybe_unused]] const bool ok = registry.apply_discovery(record.model_id, result, &updated);
    assert(ok);
    assert(updated.status == dist_model_status::manifest_pending);
    assert(updated.files.size() == 1);
    assert(updated.files[0].filename == file.filename);
    assert(updated.files[0].size_bytes == file.size_bytes);
    assert(updated.provider_revision == "main");
    assert(updated.last_discovery.time_since_epoch().count() > 0);

    [[maybe_unused]] const auto * found = registry.find(record.model_id);
    assert(found);
    assert(found->status == dist_model_status::manifest_pending);

    std::cout << "test-registry-discovery: OK\n";
    return 0;
}
