#include "orchestrator/manifest_builder/manifest_builder.h"
#include "test_manifest_common.h"

#include "gguf.h"

#include <cassert>
#include <cstdio>
#include <filesystem>
#include <string>

int main() {
    const std::string path = manifest_test_model_path();
    if (path.empty() || !std::filesystem::exists(path)) {
        fprintf(stderr, "test-gguf-header: GGUF model not found (set MODEL env)\n");
        return 77;
    }

    gguf_init_params params{};
    params.no_alloc = true;
    params.ctx      = nullptr;

    gguf_context * ctx = gguf_init_from_file(path.c_str(), params);
    assert(ctx != nullptr);

    assert(gguf_get_version(ctx) >= 2);
    assert(gguf_get_n_tensors(ctx) > 0);
    assert(gguf_get_n_kv(ctx) > 0);
    assert(gguf_get_alignment(ctx) > 0);
    assert(gguf_get_meta_size(ctx) > 24);
    assert(gguf_get_data_offset(ctx) > 0);

    gguf_free(ctx);

    printf("test-gguf-header: OK (path=%s)\n", path.c_str());
    return 0;
}
