#include "test_architecture_common.h"

#include "architecture/plugins/phi_plugin.h"

#include <cassert>
#include <cstdio>

int main() {
    phi_architecture_plugin plugin;
    const auto desc = plugin.build_descriptor(make_dense_manifest("phi3", false));

    assert(plugin.matches(make_dense_manifest("phi3", false)));
    assert(desc.family == "phi");
    assert(find_blob(desc.blobs, "embedding") != nullptr);
    assert(find_blob(desc.blobs, "output_norm") != nullptr);

    const auto rt = plugin.build_distributed_descriptor(make_dense_manifest("phi3", false));
    std::string err;
    assert(plugin.verify_runtime(rt, err));

    printf("test-phi-plugin: OK\n");
    return 0;
}
