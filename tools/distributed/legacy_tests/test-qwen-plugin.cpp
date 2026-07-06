#include "test_architecture_common.h"

#include "architecture/plugins/qwen_plugin.h"

#include <cassert>
#include <cstdio>

int main() {
    qwen_architecture_plugin plugin;
    const auto desc = plugin.build_descriptor(make_dense_manifest("qwen2", false));

    assert(desc.family == "qwen");
    assert(desc.separate_lm_head);
    const semantic_blob * head = find_blob(desc.blobs, "output_head");
    const semantic_blob * norm = find_blob(desc.blobs, "output_norm");
    assert(head && norm);
    assert(head->tensors[0].name == "output.weight");
    assert(norm->tensors[0].name == "output_norm.weight");

    printf("test-qwen-plugin: OK\n");
    return 0;
}
