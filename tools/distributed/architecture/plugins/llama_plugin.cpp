#include "architecture/plugins/llama_plugin.h"

#include "architecture/blob_builder.h"
#include "architecture/tensor_graph.h"

std::string llama_architecture_plugin::family() const {
    return "llama";
}

bool llama_architecture_plugin::matches(const model_manifest & manifest) const {
    const std::string & arch = manifest.architecture;
    if (arch_prefix_matches(arch, "qwen")) {
        return false;
    }
    if (arch_prefix_matches(arch, "gemma")) {
        return false;
    }
    return arch_prefix_matches(arch, "llama") ||
            arch_prefix_matches(arch, "tinyllama") ||
            arch_prefix_matches(arch, "mistral") ||
            arch_prefix_matches(arch, "phi") ||
            arch_prefix_matches(arch, "deepseek") ||
            arch_prefix_matches(arch, "smollm");
}

architecture_descriptor llama_architecture_plugin::build_descriptor(
        const model_manifest & manifest) const {
    architecture_descriptor desc = build_descriptor_from_graph(analyze_tensor_graph(manifest));
    desc.family = family();
    return desc;
}
