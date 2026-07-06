#include "layer_planner.h"

#include <cstdio>
#include <vector>

static bool check_plan(
        int n_layers,
        const std::vector<double> & scores,
        const std::vector<std::pair<int, int>> & expected) {
    std::vector<dist_planner_node> nodes;
    for (size_t i = 0; i < scores.size(); ++i) {
        nodes.push_back({ "node-" + std::to_string(i), scores[i] });
    }

    const auto plan = dist_plan_layers(n_layers, nodes);
    if (plan.size() != expected.size()) {
        fprintf(stderr, "size mismatch: got %zu expected %zu\n", plan.size(), expected.size());
        return false;
    }

    int covered = 0;
    for (size_t i = 0; i < plan.size(); ++i) {
        if (plan[i].layer_start != expected[i].first || plan[i].layer_end != expected[i].second) {
            fprintf(stderr, "node %zu: got [%d,%d) expected [%d,%d)\n",
                    i, plan[i].layer_start, plan[i].layer_end,
                    expected[i].first, expected[i].second);
            return false;
        }
        covered += plan[i].layer_end - plan[i].layer_start;
    }

    if (covered != n_layers) {
        fprintf(stderr, "coverage mismatch: %d != %d\n", covered, n_layers);
        return false;
    }

    if (!plan.empty() && plan.front().layer_start != 0) {
        fprintf(stderr, "plan does not start at 0\n");
        return false;
    }

    if (!plan.empty() && plan.back().layer_end != n_layers) {
        fprintf(stderr, "plan does not end at %d\n", n_layers);
        return false;
    }

    for (size_t i = 1; i < plan.size(); ++i) {
        if (plan[i].layer_start != plan[i - 1].layer_end) {
            fprintf(stderr, "gap/overlap between stage %zu and %zu\n", i - 1, i);
            return false;
        }
    }

    return true;
}

int main() {
    bool ok = true;

    ok &= check_plan(16, { 100.0 }, { { 0, 16 } });
    ok &= check_plan(16, { 100.0, 100.0 }, { { 0, 8 }, { 8, 16 } });
    ok &= check_plan(16, { 100.0, 50.0, 25.0 }, { { 0, 9 }, { 9, 14 }, { 14, 16 } });

    {
        // GPU node score >> CPU nodes: every node must still get >= 1 layer.
        const auto plan = dist_plan_layers(16, {
            { "node-c", 4385.0 }, { "node-b", 201.0 }, { "node-a", 34.0 },
        });
        if (plan.size() != 3) {
            ok = false;
        } else {
            for (const auto & p : plan) {
                if (p.layer_end <= p.layer_start) {
                    fprintf(stderr, "skewed scores: node %s got empty range\n", p.node_id.c_str());
                    ok = false;
                }
            }
        }
    }

    {
        std::vector<dist_planner_node> nodes = {
            { "a", 100.0 }, { "b", 60.0 }, { "c", 40.0 }, { "d", 20.0 },
        };
        const auto plan = dist_plan_layers(80, nodes);
        int covered = 0;
        for (size_t i = 0; i < plan.size(); ++i) {
            if (i > 0 && plan[i].layer_start != plan[i - 1].layer_end) {
                ok = false;
            }
            covered += plan[i].layer_end - plan[i].layer_start;
        }
        if (covered != 80 || plan.front().layer_start != 0 || plan.back().layer_end != 80) {
            fprintf(stderr, "4-node 80-layer plan invalid\n");
            ok = false;
        }
    }

    {
        std::vector<dist_planner_node> nodes = {
            { "a", 100.0 }, { "b", 50.0 }, { "c", 25.0 },
        };
        const auto plan = dist_plan_layers(80, nodes);
        if (plan.size() != 3) {
            ok = false;
        } else {
            const int a = plan[0].layer_end - plan[0].layer_start;
            const int b = plan[1].layer_end - plan[1].layer_start;
            const int c = plan[2].layer_end - plan[2].layer_start;
            if (a + b + c != 80) {
                ok = false;
            }
            fprintf(stderr, "80-layer split: a=%d b=%d c=%d\n", a, b, c);
        }
    }

    if (!ok) {
        fprintf(stderr, "test-layer-planner: FAILED\n");
        return 1;
    }

    printf("test-layer-planner: OK\n");
    return 0;
}
