#include "test_architecture_common.h"

#include <cassert>
#include <cstdio>

int main() {
    const auto manifest = make_dense_manifest("qwen2", false, 6);
    const auto built    = build_architecture_descriptor(manifest);

    assert(find_blob(built.blobs, "layer:0") != nullptr);
    assert(find_blob(built.blobs, "layer:5") != nullptr);
    assert(find_blob(built.blobs, "output_head") != nullptr);
    assert(find_blob(built.blobs, "output_norm") != nullptr);

    printf("test-blob-builder: OK\n");
    return 0;
}
