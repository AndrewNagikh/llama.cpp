#include "orchestrator/model_registry.h"

#include <cassert>
#include <iostream>
#include <string>
#include <thread>

static provider_discovery_result make_result(const std::string & revision) {
    provider_discovery_result result{};
    result.success    = true;
    result.provider   = "mock";
    result.repository = "mock/repo";
    result.revision   = revision;

    remote_model_file file{};
    file.filename     = "model.gguf";
    file.size_bytes   = 1000;
    result.files.push_back(file);
    return result;
}

int main() {
    cluster_model_registry registry;

    dist_model_record record{};
    record.model_id = "m1";
    record.source   = "mock";
    registry.add_or_update(record);

    const auto first = make_result("rev1");
    assert(registry.apply_discovery(record.model_id, first, nullptr));
    {
        [[maybe_unused]] const auto * r = registry.find(record.model_id);
        assert(r->provider_revision == "rev1");
        assert(r->status == dist_model_status::manifest_pending);
    }

    // Simulate the remote repository changing revision.
    const auto second = make_result("rev2");
    assert(registry.apply_discovery(record.model_id, second, nullptr));
    {
        [[maybe_unused]] const auto * r = registry.find(record.model_id);
        assert(r->provider_revision == "rev2");
        assert(r->files.size() == 1);
    }

    std::cout << "test-discovery-update: OK\n";
    return 0;
}
