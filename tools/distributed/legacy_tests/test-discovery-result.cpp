#include "orchestrator/model_provider/model_provider.h"

#include <cassert>
#include <iostream>

int main() {
    remote_model_file file{};
    file.filename     = "llama-3.2-1b-instruct-q4_k_m.gguf";
    file.size_bytes   = 807690656;
    file.etag         = "abc123";
    file.sha256       = "def456";
    file.download_url = "https://example.com/llama-3.2-1b-instruct-q4_k_m.gguf";

    provider_discovery_result result{};
    result.success    = true;
    result.provider   = "huggingface";
    result.repository = "hugging-quants/Llama-3.2-1B-Instruct-Q4_K_M-GGUF";
    result.revision   = "main";
    result.error      = "";
    result.files.push_back(file);

    const auto j = result.to_json();
    assert(j.value("success", false) == true);
    assert(j.value("provider", "") == "huggingface");
    assert(j.value("revision", "") == "main");
    assert(j["files"].is_array());
    assert(j["files"].size() == 1);
    assert(j["files"][0].value("filename", "") == file.filename);
    assert(j["files"][0].value("size_bytes", static_cast<uint64_t>(0)) == file.size_bytes);

    const auto parsed = provider_discovery_result::from_json(j);
    assert(parsed.success == result.success);
    assert(parsed.provider == result.provider);
    assert(parsed.files.size() == 1);
    assert(parsed.files[0].sha256 == file.sha256);

    std::cout << "test-discovery-result: OK\n";
    return 0;
}
