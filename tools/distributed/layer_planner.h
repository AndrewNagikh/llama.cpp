#pragma once

#include <string>
#include <vector>

struct dist_layer_assignment {
    std::string node_id;
    int         layer_start = 0;
    int         layer_end   = 0;
    double      score       = 0.0;
};

struct dist_planner_node {
    std::string node_id;
    double      score = 1.0;
};

// Proportional layer split by node score. Sorted by score descending.
// Covers [0, n_layers) exactly with no gaps or overlaps.
std::vector<dist_layer_assignment> dist_plan_layers(
        int n_layers,
        const std::vector<dist_planner_node> & nodes);
