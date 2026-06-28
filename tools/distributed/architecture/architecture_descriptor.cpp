#include "architecture_descriptor.h"

#include "architecture_plugin.h"
#include "tensor_graph.h"

#include <set>

std::vector<std::string> nodes_for_blob_deploy(
        const blob_deploy_target deploy,
        const std::string & entry_node,
        const std::string & final_node,
        const std::vector<std::string> & all_nodes) {
    std::vector<std::string> nodes;
    switch (deploy) {
        case blob_deploy_target::entry_node:
            if (!entry_node.empty()) {
                nodes.push_back(entry_node);
            }
            break;
        case blob_deploy_target::final_node:
            if (!final_node.empty()) {
                nodes.push_back(final_node);
            }
            break;
        case blob_deploy_target::all_nodes:
            nodes = all_nodes;
            break;
        case blob_deploy_target::none:
            break;
    }
    return nodes;
}

architecture_descriptor build_architecture_descriptor(const model_manifest & manifest) {
    const tensor_graph graph = analyze_tensor_graph(manifest);
    architecture_descriptor desc = build_descriptor_from_graph(graph);
    desc.family = select_architecture_plugin(manifest).family();
    return desc;
}
