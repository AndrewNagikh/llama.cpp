#include "architecture_descriptor.h"

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
