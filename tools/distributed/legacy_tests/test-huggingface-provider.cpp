#include "orchestrator/model_provider/huggingface_provider.h"
#include "orchestrator/model_registry.h"

#include <cassert>
#include <iostream>
#include <string>

int main() {
    auto provider = create_huggingface_provider();
    assert(provider);
    assert(provider->name() == "huggingface");

    dist_model_record model{};
    model.model_id   = "llama-3.2-1b";
    model.source     = "huggingface";
    model.repository = "hugging-quants/Llama-3.2-1B-Instruct-Q4_K_M-GGUF";
    model.revision   = "main";

    const auto result = provider->discover(model);
    if (!result.success) {
        std::cerr << "test-huggingface-provider: discovery failed: " << result.error << "\n";
        return 1;
    }

    assert(result.provider == "huggingface");
    assert(result.repository == model.repository);
    assert(result.revision == model.revision);
    assert(!result.files.empty());

    [[maybe_unused]] bool found_gguf = false;
    for (const auto & file : result.files) {
        assert(!file.filename.empty());
        assert(file.size_bytes > 0);
        assert(!file.download_url.empty());
        assert(file.download_url.find("huggingface.co") != std::string::npos);
        if (file.filename == "llama-3.2-1b-instruct-q4_k_m.gguf") {
            found_gguf = true;
            std::cout << "test-huggingface-provider: found " << file.filename
                      << " size=" << file.size_bytes << "\n";
        }
    }
    assert(found_gguf);

    std::cout << "test-huggingface-provider: OK (files=" << result.files.size() << ")\n";
    return 0;
}
