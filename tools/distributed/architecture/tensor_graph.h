#pragma once

#include "architecture_descriptor.h"

#include "orchestrator/manifest_builder/manifest_builder.h"

// Research-first manifest analysis (Task 9.9).
// Classifies every tensor and builds semantic blobs + worker requirements
// without architecture-specific conditionals in runtime code.

struct classified_tensor {
    tensor_descriptor     tensor;
    tensor_semantic_role  semantic = tensor_semantic_role::unknown;
    int32_t               layer_index = -1;
};

struct tensor_graph {
    std::string                  architecture;
    std::vector<classified_tensor> tensors;
};

tensor_graph analyze_tensor_graph(const model_manifest & manifest);

architecture_descriptor build_descriptor_from_graph(const tensor_graph & graph);
