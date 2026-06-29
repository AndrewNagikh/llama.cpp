#include "architecture/plugins/deepseek_plugin.h"

#include "architecture/blob_builder.h"
#include "architecture/tensor_graph.h"

std::string deepseek_architecture_plugin::family() const {
    return "deepseek";
}

bool deepseek_architecture_plugin::matches(const model_manifest & manifest) const {
    return arch_prefix_matches(manifest.architecture, "deepseek");
}

architecture_descriptor deepseek_architecture_plugin::build_descriptor(
        const model_manifest & manifest) const {
    architecture_descriptor desc = build_descriptor_from_graph(analyze_tensor_graph(manifest));
    desc.family = family();
    return desc;
}
