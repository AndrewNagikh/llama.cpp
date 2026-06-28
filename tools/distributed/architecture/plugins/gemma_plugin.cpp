#include "architecture/plugins/gemma_plugin.h"

#include "architecture/blob_builder.h"
#include "architecture/tensor_graph.h"

std::string gemma_architecture_plugin::family() const {
    return "gemma";
}

bool gemma_architecture_plugin::matches(const model_manifest & manifest) const {
    return arch_prefix_matches(manifest.architecture, "gemma");
}

architecture_descriptor gemma_architecture_plugin::build_descriptor(
        const model_manifest & manifest) const {
    architecture_descriptor desc = build_descriptor_from_graph(analyze_tensor_graph(manifest));
    desc.family = family();
    return desc;
}
