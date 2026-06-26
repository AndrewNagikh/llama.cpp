#include "architecture_descriptor/architecture_descriptor.h"

#include <cassert>
#include <cstdio>

int main() {
    int32_t layer = -1;
    tensor_role role = tensor_role::unknown;

    classify_tensor_enhanced("token_embd.weight", layer, role);
    assert(role == tensor_role::embedding);
    assert(classify_tensor_semantic("token_embd.weight", role, layer) ==
            tensor_semantic_role::embedding);

    classify_tensor_enhanced("output.weight", layer, role);
    assert(role == tensor_role::lm_head);
    assert(classify_tensor_semantic("output.weight", role, layer) ==
            tensor_semantic_role::output_head);

    classify_tensor_enhanced("blk.3.ffn_gate_inp.weight", layer, role);
    assert(layer == 3);
    assert(classify_tensor_semantic("blk.3.ffn_gate_inp.weight", role, layer) ==
            tensor_semantic_role::router);

    classify_tensor_enhanced("blk.1.ffn_down_exps.weight", layer, role);
    assert(classify_tensor_semantic("blk.1.ffn_down_exps.weight", role, layer) ==
            tensor_semantic_role::expert);

    classify_tensor_enhanced("blk.0.attn_q.weight", layer, role);
    assert(classify_tensor_semantic("blk.0.attn_q.weight", role, layer) ==
            tensor_semantic_role::attention);

    printf("test-semantic-roles: OK\n");
    return 0;
}
