#include "architecture/plugins/smol_plugin.h"

#include "architecture/blob_builder.h"
#include "architecture/tensor_graph.h"

std::string smol_architecture_plugin::family() const {
    return "smollm";
}

bool smol_architecture_plugin::matches(const model_manifest & manifest) const {
    const std::string & arch = manifest.architecture;
    return arch_prefix_matches(arch, "smollm") || arch_prefix_matches(arch, "smol");
}

architecture_descriptor smol_architecture_plugin::build_descriptor(
        const model_manifest & manifest) const {
    architecture_descriptor desc = build_descriptor_from_graph(analyze_tensor_graph(manifest));
    desc.family = family();
    return desc;
}
