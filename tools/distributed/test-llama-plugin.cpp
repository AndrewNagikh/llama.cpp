#include "test_architecture_common.h"

#include "architecture/architecture_plugin.h"
#include "architecture/plugins/llama_plugin.h"

#include <cassert>
#include <cstdio>

int main() {
    llama_architecture_plugin plugin;
    const auto tied = plugin.build_descriptor(make_dense_manifest("llama", true));
    assert(tied.family == "llama");
    assert(tied.tied_embeddings);
    assert(find_blob(tied.blobs, "embedding")->deploy == blob_deploy_target::all_nodes);

    printf("test-llama-plugin: OK\n");
    return 0;
}
