#include "orchestrator/model_provider/huggingface_provider.h"
#include "orchestrator/model_provider/model_provider.h"
#include "orchestrator/model_registry.h"

#include <cassert>
#include <iostream>
#include <memory>
#include <string>

int main() {
    // Unknown provider factory returns nullptr.
    std::unique_ptr<model_provider> unknown = create_model_provider("unknown");
    assert(!unknown);
    std::cout << "test-discovery-errors: unknown provider correctly rejected\n";

    // Missing repository should fail gracefully.
    auto hf = create_huggingface_provider();
    assert(hf);

    dist_model_record bad_model{};
    bad_model.model_id   = "missing";
    bad_model.source     = "huggingface";
    bad_model.repository = "this-repo-does-not-exist-42b/zzz";
    bad_model.revision   = "main";

    const auto bad_result = hf->discover(bad_model);
    assert(!bad_result.success);
    std::cout << "test-discovery-errors: missing repo error: " << bad_result.error << "\n";

    // Empty repository should fail without a network call.
    dist_model_record empty_repo{};
    empty_repo.model_id = "empty";
    empty_repo.source   = "huggingface";

    const auto empty_result = hf->discover(empty_repo);
    assert(!empty_result.success);

    std::cout << "test-discovery-errors: OK\n";
    return 0;
}
