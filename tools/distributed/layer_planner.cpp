#include "layer_planner.h"

#include "dist_common.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

struct dist_frac_part {
    size_t idx;
    double remainder;
};

static void dist_plan_fill_proportional(
        int n_layers,
        std::vector<double> & scores,
        std::vector<int> & counts) {
    const size_t n_nodes = scores.size();
    counts.assign(n_nodes, 0);
    if (n_nodes == 0) {
        return;
    }

    double total_score = 0.0;
    for (double s : scores) {
        total_score += s > 0.0 ? s : 1.0;
    }
    if (total_score <= 0.0) {
        total_score = static_cast<double>(n_nodes);
    }

    std::vector<dist_frac_part> remainders;
    remainders.reserve(n_nodes);

    int assigned = 0;
    for (size_t i = 0; i < n_nodes; ++i) {
        const double s = scores[i] > 0.0 ? scores[i] : 1.0;
        const double exact = static_cast<double>(n_layers) * s / total_score;
        counts[i] = static_cast<int>(std::floor(exact));
        assigned += counts[i];
        remainders.push_back({ i, exact - static_cast<double>(counts[i]) });
    }

    int leftover = n_layers - assigned;
    std::sort(remainders.begin(), remainders.end(), [](const dist_frac_part & a, const dist_frac_part & b) {
        if (a.remainder != b.remainder) {
            return a.remainder > b.remainder;
        }
        return a.idx < b.idx;
    });

    for (int i = 0; i < leftover && i < static_cast<int>(n_nodes); ++i) {
        counts[remainders[(size_t) i].idx]++;
    }
}

// Existing score-only planner (Task 6). Kept for compatibility.
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

    std::vector<double> scores(n_nodes);
    for (size_t i = 0; i < n_nodes; ++i) {
        scores[i] = sorted[i].score;
    }
    std::vector<int> counts(n_nodes, 0);
    dist_plan_fill_proportional(n_layers, scores, counts);

    // Each node must run at least one layer when the model has enough layers.
    if (n_layers >= static_cast<int>(n_nodes)) {
        for (size_t i = 0; i < n_nodes; ++i) {
            if (counts[i] > 0) {
                continue;
            }
            size_t donor = 0;
            for (size_t j = 1; j < n_nodes; ++j) {
                if (counts[j] > counts[donor]) {
                    donor = j;
                }
            }
            if (counts[donor] <= 1) {
                break;
            }
            counts[donor]--;
            counts[i]++;
        }
    }

    int cursor = 0;
    for (size_t i = 0; i < n_nodes; ++i) {
        out[i].node_id     = sorted[i].node_id;
        out[i].score       = sorted[i].score;
        out[i].layer_start = cursor;
        out[i].layer_end   = cursor + counts[i];
        out[i].device_hint = "cpu";
        cursor += counts[i];
    }

    return out;
}

// ---------------------------------------------------------------------------
// Task 8: memory-aware planner
// ---------------------------------------------------------------------------

struct planner_internal_node {
    size_t orig_idx = 0;
    std::string node_id;
    double score = 1.0;
    std::string backend;
    uint64_t budget_bytes = 0;
    bool has_gpu = false;
    int max_layers = 0;
    int count = 0;
};

static uint64_t dist_layer_total_bytes(
        const model_memory_requirements & mem,
        size_t layer_idx) {
    const uint64_t overhead = mem.kv_bytes + mem.compute_bytes + mem.scratch_bytes;
    const uint64_t per_layer_overhead = mem.n_layer > 0
            ? overhead / static_cast<uint64_t>(mem.n_layer)
            : 0;
    if (layer_idx >= mem.layers.size()) {
        return per_layer_overhead;
    }
    return mem.layers[layer_idx].weight_bytes + per_layer_overhead;
}

dist_planner_result dist_plan_layers_memory_aware(
        const model_memory_requirements & mem,
        const std::vector<dist_planner_node_resources> & nodes) {
    dist_planner_result result{};

    if (mem.n_layer <= 0 || mem.layers.empty()) {
        result.error = "invalid model memory requirements";
        return result;
    }
    if (nodes.empty()) {
        result.error = "no nodes available";
        return result;
    }

    const int n_layers = mem.n_layer;
    const uint64_t total_bytes = mem.total_bytes();
    if (total_bytes == 0) {
        result.error = "model has zero byte memory estimate";
        return result;
    }

    std::vector<planner_internal_node> work;
    work.reserve(nodes.size());
    for (size_t i = 0; i < nodes.size(); ++i) {
        const auto & n = nodes[i];
        planner_internal_node p{};
        p.orig_idx     = i;
        p.node_id      = n.node_id;
        p.score        = n.score > 0.0 ? n.score : 1.0;
        p.backend      = n.backend;
        p.has_gpu      = n.has_gpu;
        // Primary execution budget: GPU VRAM if present, otherwise CPU RAM.
        p.budget_bytes = n.has_gpu ? n.gpu_budget_bytes : n.cpu_budget_bytes;
        work.push_back(p);
    }

    // Sort by score descending, then allocate layer counts proportional to
    // score while capping each node to its primary memory budget.  This keeps
    // multi-node layouts when memory allows and avoids the pre-existing single-
    // stage pipeline edge case, while still skipping nodes that cannot hold a
    // single layer.
    std::sort(work.begin(), work.end(), [](const planner_internal_node & a, const planner_internal_node & b) {
        if (a.score != b.score) {
            return a.score > b.score;
        }
        return a.node_id < b.node_id;
    });

    const size_t n_nodes = work.size();

    // Compute per-node capacity in layers, measured against the layers starting
    // from the current model head.  Uniform layers make this exact; for
    // heterogeneous layer sizes this is a conservative per-node cap.
    int cap_cursor = 0;
    std::vector<int> caps(n_nodes, 0);
    for (size_t i = 0; i < n_nodes; ++i) {
        const auto & w = work[i];
        if (w.budget_bytes == 0) {
            continue;
        }
        uint64_t used = 0;
        int local_cursor = cap_cursor;
        while (local_cursor < n_layers) {
            const uint64_t cost = dist_layer_total_bytes(mem, static_cast<size_t>(local_cursor));
            if (cost == 0) {
                break;
            }
            if (cost > w.budget_bytes || used + cost > w.budget_bytes) {
                break;
            }
            used += cost;
            ++local_cursor;
        }
        caps[i] = local_cursor - cap_cursor;
    }

    std::vector<size_t> active;
    for (size_t i = 0; i < n_nodes; ++i) {
        if (caps[i] > 0) {
            active.push_back(i);
        }
    }
    if (active.empty()) {
        result.error = "model does not fit: no node can hold even one layer";
        return result;
    }

    // Proportional allocation among active nodes.
    std::vector<double> active_scores;
    active_scores.reserve(active.size());
    for (size_t idx : active) {
        active_scores.push_back(work[idx].score);
    }
    std::vector<int> active_counts(active.size(), 0);
    dist_plan_fill_proportional(n_layers, active_scores, active_counts);

    std::vector<int> counts(n_nodes, 0);
    for (size_t k = 0; k < active.size(); ++k) {
        counts[active[k]] = active_counts[k];
    }

    // Make sure every active node receives at least one layer when possible.
    if (n_layers >= static_cast<int>(active.size())) {
        for (size_t k = 0; k < active.size(); ++k) {
            const size_t i = active[k];
            if (counts[i] > 0) {
                continue;
            }
            size_t donor = active[0];
            for (size_t j = 1; j < active.size(); ++j) {
                const size_t cand = active[j];
                if (counts[cand] > counts[donor] && counts[cand] > 1) {
                    donor = cand;
                }
            }
            if (counts[donor] <= 1) {
                continue;
            }
            counts[donor]--;
            counts[i]++;
        }
    }

    // Clamp any over-cap node and redistribute freed layers.
    for (int iter = 0; iter < static_cast<int>(active.size()) + 1; ++iter) {
        int excess = 0;
        double slack_score_total = 0.0;

        for (size_t i : active) {
            if (counts[i] > caps[i]) {
                excess += counts[i] - caps[i];
                counts[i] = caps[i];
            }
            if (counts[i] < caps[i]) {
                slack_score_total += work[i].score;
            }
        }
        if (excess == 0) {
            break;
        }

        // Distribute proportionally by score, never overfilling.
        int remaining = excess;
        for (size_t i : active) {
            if (counts[i] >= caps[i] || slack_score_total <= 0.0) {
                continue;
            }
            const int share = std::max(1, static_cast<int>(std::round(
                    static_cast<double>(excess) * work[i].score / slack_score_total)));
            const int room = caps[i] - counts[i];
            const int add = std::min({share, remaining, room});
            counts[i] += add;
            remaining -= add;
        }

        for (size_t i : active) {
            if (remaining <= 0) {
                break;
            }
            if (counts[i] >= caps[i]) {
                continue;
            }
            const int room = caps[i] - counts[i];
            const int add = std::min(room, remaining);
            counts[i] += add;
            remaining -= add;
        }

        if (remaining > 0) {
            result.error = "model does not fit: cannot redistribute layers within cluster budgets";
            return result;
        }
    }

    int total_assigned = 0;
    for (int c : counts) {
        total_assigned += c;
    }
    if (total_assigned != n_layers) {
        result.error = "model does not fit: assigned " + std::to_string(total_assigned) +
                      "/" + std::to_string(n_layers) + " layers";
        return result;
    }

    int cursor = 0;
    for (size_t i = 0; i < n_nodes; ++i) {
        if (counts[i] <= 0) {
            continue;
        }
        dist_layer_assignment assign{};
        assign.node_id     = work[i].node_id;
        assign.score       = work[i].score;
        assign.layer_start = cursor;
        assign.layer_end   = cursor + counts[i];
        assign.device_hint = work[i].has_gpu ? "gpu" : "cpu";
        result.assignments.push_back(assign);
        cursor += counts[i];
    }

    if (cursor != n_layers) {
        result.assignments.clear();
        result.error = "model does not fit: layer coverage incomplete";
        return result;
    }

    result.success = true;
    return result;
}
