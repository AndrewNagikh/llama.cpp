#include "architecture/plugins/phi_plugin.h"

#include "architecture/blob_builder.h"
#include "architecture/tensor_graph.h"

std::string phi_architecture_plugin::family() const {
    return "phi";
}

bool phi_architecture_plugin::matches(const model_manifest & manifest) const {
    return arch_prefix_matches(manifest.architecture, "phi");
}

architecture_descriptor phi_architecture_plugin::build_descriptor(
        const model_manifest & manifest) const {
    architecture_descriptor desc = build_descriptor_from_graph(analyze_tensor_graph(manifest));
    desc.family = family();
    return desc;
}
