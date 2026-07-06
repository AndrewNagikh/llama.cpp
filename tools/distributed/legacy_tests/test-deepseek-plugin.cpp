#include "test_architecture_common.h"

#include "architecture/plugins/deepseek_plugin.h"

#include <cassert>
#include <cstdio>

int main() {
    deepseek_architecture_plugin plugin;
    const auto desc = plugin.build_descriptor(make_dense_manifest("deepseek", false));

    assert(plugin.matches(make_dense_manifest("deepseek", false)));
    assert(desc.family == "deepseek");
    assert(find_blob(desc.blobs, "output_head") != nullptr || desc.separate_lm_head);

    const auto rt = plugin.build_distributed_descriptor(make_dense_manifest("deepseek", false));
    std::string err;
    assert(plugin.verify_runtime(rt, err));

    printf("test-deepseek-plugin: OK\n");
    return 0;
}
