#include "test_architecture_common.h"

#include "architecture/plugins/gemma_plugin.h"

#include <cassert>
#include <cstdio>

int main() {
    gemma_architecture_plugin plugin;
    const auto tied = plugin.build_descriptor(make_dense_manifest("gemma2", true));
    assert(tied.family == "gemma");

    const auto untied = plugin.build_descriptor(make_dense_manifest("gemma2", false));
    assert(untied.separate_lm_head);

    printf("test-gemma-plugin: OK\n");
    return 0;
}
