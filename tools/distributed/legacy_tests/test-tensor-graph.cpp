#include "test_architecture_common.h"

#include "architecture/tensor_graph.h"

#include <cassert>
#include <cstdio>

int main() {
    const model_manifest manifest = make_dense_manifest("qwen2", true, 8);
    const tensor_graph graph = analyze_tensor_graph(manifest);
    assert(graph.tensors.size() == manifest.tensors.size());

    const architecture_descriptor desc = build_descriptor_from_graph(graph);
    assert(desc.tied_embeddings);
    assert(find_blob(desc.blobs, "embedding") != nullptr);
    assert(find_blob(desc.blobs, "layer:0") != nullptr);
    assert(find_blob(desc.blobs, "layer:7") != nullptr);

    const semantic_blob * head = find_blob(desc.blobs, "output_head");
    assert(head != nullptr);
    assert(head->storage_alias);

    printf("test-tensor-graph: OK tensors=%zu blobs=%zu\n",
            graph.tensors.size(), desc.blobs.size());
    return 0;
}
