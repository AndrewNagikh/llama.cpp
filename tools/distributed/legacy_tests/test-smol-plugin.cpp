#include "test_architecture_common.h"

#include "architecture/plugins/smol_plugin.h"

#include <cassert>
#include <cstdio>

int main() {
    smol_architecture_plugin plugin;
    const auto desc = plugin.build_descriptor(make_dense_manifest("smollm", true));

    assert(plugin.matches(make_dense_manifest("smollm", true)));
    assert(desc.family == "smollm");
    assert(find_blob(desc.blobs, "embedding") != nullptr);

    const auto rt = plugin.build_distributed_descriptor(make_dense_manifest("smollm", true));
    std::string err;
    assert(plugin.verify_runtime(rt, err));

    printf("test-smol-plugin: OK\n");
    return 0;
}
