#include "orchestrator/manifest_builder/manifest_builder.h"

#include <cassert>
#include <cstdio>
#include <string>
#include <vector>

int main() {
  {
    std::vector<uint8_t> bad = { 'B', 'A', 'D', '!' };
    std::string error;
    const model_manifest manifest = build_manifest_from_buffer(bad, error);
    assert(manifest.empty());
    assert(!error.empty());
    assert(error.find("magic") != std::string::npos);
  }

  {
    std::vector<uint8_t> truncated = { 'G', 'G', 'U', 'F' };
    std::string error;
    const model_manifest manifest = build_manifest_from_buffer(truncated, error);
    assert(manifest.empty());
    assert(!error.empty());
  }

    printf("test-corrupted-header: OK\n");
    return 0;
}
