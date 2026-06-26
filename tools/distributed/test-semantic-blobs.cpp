#include "test_architecture_common.h"

#include <cassert>
#include <cstdio>

int main() {
    const auto qwen = build_architecture_descriptor(make_dense_manifest("qwen2", false));
    const semantic_blob * head = find_blob(qwen.blobs, "output_head");
    const semantic_blob * norm = find_blob(qwen.blobs, "output_norm");

    assert(head != nullptr);
    assert(norm != nullptr);
    assert(head->tensors.size() == 1);
    assert(norm->tensors.size() == 1);
    assert(head->tensors[0].offset != norm->tensors[0].offset);

    printf("test-semantic-blobs: OK\n");
    return 0;
}
