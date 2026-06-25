#include "orchestrator/model_provider/model_provider.h"
#include "orchestrator/model_registry.h"

#include <cassert>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

namespace {

class mock_provider : public model_provider {
public:
    std::string name() const override { return "mock"; }

    provider_discovery_result discover(const dist_model_record & /*model*/) override {
        provider_discovery_result result{};
        result.success   = true;
        result.provider  = "mock";
        result.repository = "mock/repo";
        result.revision  = "v1";

        remote_model_file file{};
        file.filename    = "model.gguf";
        file.size_bytes  = 12345;
        file.etag        = "etag";
        file.sha256      = "sha256";
        file.download_url = "https://example.com/model.gguf";
        result.files.push_back(file);

        return result;
    }
};

} // namespace

int main() {
    std::unique_ptr<model_provider> provider = std::make_unique<mock_provider>();
    assert(provider);
    assert(provider->name() == "mock");

    dist_model_record model{};
    model.model_id   = "m1";
    model.repository = "mock/repo";
    model.revision   = "v1";

    const auto result = provider->discover(model);
    assert(result.success);
    assert(result.provider == "mock");
    assert(result.files.size() == 1);
    assert(result.files[0].filename == "model.gguf");
    assert(result.files[0].size_bytes == 12345);

    std::cout << "test-provider-interface: OK\n";
    return 0;
}
