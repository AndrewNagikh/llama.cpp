#include "layer_planner.h"

#include <algorithm>
#include <cmath>
#include <vector>

struct dist_frac_part {
    size_t idx;
    double remainder;
};

std::vector<dist_layer_assignment> dist_plan_layers(
        int n_layers,
        const std::vector<dist_planner_node> & nodes) {
    std::vector<dist_layer_assignment> out;
    if (n_layers <= 0 || nodes.empty()) {
        return out;
    }

    auto sorted = nodes;
    std::sort(sorted.begin(), sorted.end(), [](const dist_planner_node & a, const dist_planner_node & b) {
        if (a.score != b.score) {
            return a.score > b.score;
        }
        return a.node_id < b.node_id;
    });

    const size_t n_nodes = sorted.size();
    out.resize(n_nodes);

    if (n_nodes == 1) {
        out[0].node_id     = sorted[0].node_id;
        out[0].score       = sorted[0].score;
        out[0].layer_start = 0;
        out[0].layer_end   = n_layers;
        return out;
    }

    double total_score = 0.0;
    for (const auto & node : sorted) {
        total_score += node.score > 0.0 ? node.score : 1.0;
    }
    if (total_score <= 0.0) {
        total_score = (double) n_nodes;
        for (auto & node : sorted) {
            (void) node;
        }
    }

    std::vector<int> counts(n_nodes, 0);
    std::vector<dist_frac_part> remainders;
    remainders.reserve(n_nodes);

    int assigned = 0;
    for (size_t i = 0; i < n_nodes; ++i) {
        const double score = sorted[i].score > 0.0 ? sorted[i].score : 1.0;
        const double exact = (double) n_layers * score / total_score;
        counts[i] = (int) std::floor(exact);
        assigned += counts[i];
        remainders.push_back({ i, exact - (double) counts[i] });
    }

    int leftover = n_layers - assigned;
    std::sort(remainders.begin(), remainders.end(), [](const dist_frac_part & a, const dist_frac_part & b) {
        if (a.remainder != b.remainder) {
            return a.remainder > b.remainder;
        }
        return a.idx < b.idx;
    });

    for (int i = 0; i < leftover; ++i) {
        counts[remainders[(size_t) i].idx]++;
    }

    int cursor = 0;
    for (size_t i = 0; i < n_nodes; ++i) {
        out[i].node_id     = sorted[i].node_id;
        out[i].score       = sorted[i].score;
        out[i].layer_start = cursor;
        out[i].layer_end   = cursor + counts[i];
        cursor += counts[i];
    }

    return out;
}
