#include "layout_planner.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <map>
#include <set>

using json = nlohmann::json;

static constexpr uint64_t MIN_COMPUTE_BYTES = 256ULL * 1024ULL * 1024ULL;
static constexpr uint64_t MIN_SCRATCH_BYTES = 128ULL * 1024ULL * 1024ULL;
static constexpr uint64_t KV_ELEMENT_BYTES = 2;

// ---------------------------------------------------------------------------
// JSON
// ---------------------------------------------------------------------------

json layer_placement::to_json() const {
    return {
        { "layer", layer_index },
        { "node", node_id },
        { "device", device },
        { "size_bytes", size_bytes },
        { "size_mb", static_cast<int64_t>(size_bytes / (1024 * 1024)) },
        { "required", required },
    };
}

layer_placement layer_placement::from_json(const json & j) {
    layer_placement p;
    p.layer_index = j.value("layer", j.value("layer_index", -1));
    p.node_id     = j.value("node", j.value("node_id", ""));
    p.device      = j.value("device", "");
    p.size_bytes  = j.value("size_bytes", static_cast<uint64_t>(0));
    if (p.size_bytes == 0 && j.contains("size_mb")) {
        p.size_bytes = static_cast<uint64_t>(j.value("size_mb", 0)) * 1024 * 1024;
    }
    p.required = j.value("required", true);
    return p;
}

json desired_model_layout::to_json() const {
    json placements_json = json::array();
    for (const auto & p : placements) {
        placements_json.push_back(p.to_json());
    }
    json warnings_json = json::array();
    for (const auto & w : warnings) {
        warnings_json.push_back(w);
    }
    return {
        { "model", model_id },
        { "model_id", model_id },
        { "fits_cluster", fits_cluster },
        { "placements", placements_json },
        { "total_weight_bytes", total_weight_bytes },
        { "total_required_memory", total_required_memory },
        { "warnings", warnings_json },
    };
}

desired_model_layout desired_model_layout::from_json(const json & j) {
    desired_model_layout layout;
    layout.model_id = j.value("model_id", j.value("model", ""));
    layout.fits_cluster = j.value("fits_cluster", false);
    layout.total_weight_bytes = j.value("total_weight_bytes", static_cast<uint64_t>(0));
    layout.total_required_memory = j.value("total_required_memory", static_cast<uint64_t>(0));
    if (j.contains("placements") && j["placements"].is_array()) {
        for (const auto & item : j["placements"]) {
            layout.placements.push_back(layer_placement::from_json(item));
        }
    }
    if (j.contains("warnings") && j["warnings"].is_array()) {
        for (const auto & w : j["warnings"]) {
            layout.warnings.push_back(w.get<std::string>());
        }
    }
    return layout;
}

json model_layout::to_json() const {
    return desired.to_json();
}

model_layout model_layout::from_json(const json & j) {
    model_layout ml;
    ml.desired = desired_model_layout::from_json(j);
    return ml;
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

layout_node_input layout_node_from_dist(const dist_node_info & node) {
    layout_node_input n;
    n.node_id           = node.node_id;
    n.score             = node.score > 0.0 ? node.score : node.performance.score;
    n.backend           = node.caps.gpu_backend.empty() ? "cpu" : node.caps.gpu_backend;
    n.has_gpu           = node.memory.has_gpu;
    n.cpu_budget_bytes  = node.memory.free_ram_bytes;
    n.gpu_budget_bytes  = node.memory.free_vram_bytes;
    n.decode_tps        = node.performance.decode_tps;
    return n;
}

std::string layout_normalize_device(const layout_node_input & node) {
    return dist_normalize_device(node.backend, node.has_gpu);
}

model_memory_requirements memory_requirements_from_manifest(
        const model_manifest & manifest,
        int32_t n_ctx) {
    model_memory_requirements mem{};
    mem.model_id = manifest.architecture.empty() ? "unknown" : manifest.architecture;
    mem.n_layer  = static_cast<int32_t>(manifest.layers.size());
    if (manifest.n_layer > 0) {
        mem.n_layer = static_cast<int32_t>(manifest.n_layer);
    }
    mem.n_embd = manifest.n_embd > 0 ? static_cast<int32_t>(manifest.n_embd) : 1;
    mem.n_ctx  = n_ctx > 0 ? n_ctx : (manifest.n_ctx > 0 ? static_cast<int32_t>(manifest.n_ctx) : 4096);

    mem.layers.reserve(manifest.layers.size());
    for (const auto & ld : manifest.layers) {
        model_layer_memory layer{};
        layer.layer_index  = ld.layer_index;
        layer.weight_bytes = ld.size_bytes;
        mem.layers.push_back(layer);
        mem.weights_bytes += ld.size_bytes;
    }

    if (mem.weights_bytes == 0) {
        for (const auto & t : manifest.tensors) {
            if (t.layer >= 0) {
                mem.weights_bytes += t.size_bytes;
            }
        }
    }

    const uint64_t kv_dim_per_token = static_cast<uint64_t>(mem.n_embd) / 4;
    const uint64_t kv_per_layer_for_ctx = 2 * kv_dim_per_token * KV_ELEMENT_BYTES *
                                          static_cast<uint64_t>(mem.n_ctx);
    mem.kv_bytes = kv_per_layer_for_ctx * static_cast<uint64_t>(std::max(1, mem.n_layer));
    // MoE-aware safety margin (2026-07-23 fix, see memory_estimator.cpp for
    // the full incident writeup): this is the ACTUAL function build_desired_
    // layout() calls, so this is where the fix has to land, not just the
    // duplicate estimate_model_memory_from_manifest() in memory_estimator.cpp.
    const bool is_moe = model_architecture_is_moe(manifest.architecture);
    mem.compute_bytes = std::max(MIN_COMPUTE_BYTES, mem.weights_bytes / (is_moe ? 4 : 16));
    mem.scratch_bytes = std::max(MIN_SCRATCH_BYTES, mem.weights_bytes / (is_moe ? 8 : 32));
    return mem;
}

namespace {

struct layout_internal_node {
    layout_node_input input;
    std::string device;
    uint64_t budget_bytes = 0;
};

static uint64_t layer_placement_cost(
        const model_memory_requirements & mem,
        int32_t layer_index) {
    const uint64_t overhead = mem.kv_bytes + mem.compute_bytes + mem.scratch_bytes;
    const uint64_t per_layer_overhead = mem.n_layer > 0
            ? overhead / static_cast<uint64_t>(mem.n_layer)
            : 0;
    if (layer_index < 0 || layer_index >= static_cast<int32_t>(mem.layers.size())) {
        return per_layer_overhead;
    }
    return mem.layers[static_cast<size_t>(layer_index)].weight_bytes + per_layer_overhead;
}

static uint64_t manifest_layer_weight_bytes(const model_manifest & manifest, int32_t layer_index) {
    for (const auto & ld : manifest.layers) {
        if (ld.layer_index == layer_index) {
            return ld.size_bytes;
        }
    }
    return 0;
}

static bool node_sort_order(const layout_internal_node & a, const layout_internal_node & b) {
    const bool a_gpu = a.input.has_gpu;
    const bool b_gpu = b.input.has_gpu;
    if (a_gpu != b_gpu) {
        return a_gpu > b_gpu;
    }
    if (a.input.score != b.input.score) {
        return a.input.score > b.input.score;
    }
    return a.input.node_id < b.input.node_id;
}

// Pipeline order of a stored layout: nodes by first appearance walking
// placements by ascending layer index, i.e. entry, middle..., final.
static std::vector<std::string> stage_order_from_layout(const desired_model_layout & layout) {
    std::vector<const layer_placement *> sorted;
    sorted.reserve(layout.placements.size());
    for (const auto & p : layout.placements) {
        sorted.push_back(&p);
    }
    std::sort(sorted.begin(), sorted.end(),
            [](const layer_placement * a, const layer_placement * b) {
                return a->layer_index < b->layer_index;
            });

    std::vector<std::string> order;
    std::set<std::string> seen;
    for (const auto * p : sorted) {
        if (seen.insert(p->node_id).second) {
            order.push_back(p->node_id);
        }
    }
    return order;
}

// Invert the stage assignment back into the score ordering it came from.
// build_desired_layout lays stages out as [active[1], active[2], ...,
// active[N-1], active[0]] -- final (heaviest stage) to the strongest node,
// entry to the second strongest -- so a stored layout read back in pipeline
// order has to be un-rotated before it can be compared against a freshly
// score-sorted `active`.
static std::vector<std::string> score_order_from_layout(const desired_model_layout & layout) {
    const std::vector<std::string> stages = stage_order_from_layout(layout);
    if (stages.size() < 2) {
        return stages;
    }
    std::vector<std::string> scored;
    scored.reserve(stages.size());
    scored.push_back(stages.back());   // final was active[0]
    scored.push_back(stages.front());  // entry was active[1]
    for (size_t i = 1; i + 1 < stages.size(); ++i) {
        scored.push_back(stages[i]);   // middles were active[2..]
    }
    return scored;
}

// Relayout hysteresis. `active` arrives sorted by the current (noisy) scores;
// if the previous layout used a different ordering of the very same nodes,
// keep the previous one unless the node that would take the more important
// role is meaningfully -- not marginally -- stronger.
//
// Applied as a single decision over the whole ordering rather than as a
// tie-break inside node_sort_order: a pairwise "keep previous unless better
// by X%" comparator is not a strict weak ordering (scores 100/105/111 at 10%
// yield A<B, B<C, C<A), and feeding that to std::sort is undefined behavior.
static void apply_role_hysteresis(
        std::vector<layout_internal_node *> & active,
        const desired_model_layout * previous) {
    if (previous == nullptr || previous->placements.empty() || active.size() < 2) {
        return;
    }

    std::map<std::string, layout_internal_node *> by_id;
    for (auto * w : active) {
        by_id[w->input.node_id] = w;
    }

    // Build the previous ordering, restricted to nodes that are still active.
    std::vector<layout_internal_node *> prev_order;
    prev_order.reserve(active.size());
    for (const auto & node_id : score_order_from_layout(*previous)) {
        const auto it = by_id.find(node_id);
        if (it != by_id.end()) {
            prev_order.push_back(it->second);
        }
    }

    // Only meaningful when the previous layout covers exactly today's active
    // set. A node that joined or dropped out is a real topology change, not
    // score noise -- take the fresh ordering.
    if (prev_order.size() != active.size()) {
        return;
    }

    // Never let hysteresis pull a GPU node behind a CPU-only one; that
    // ordering rule reflects hardware, not a measured score.
    bool seen_non_gpu = false;
    for (const auto * w : prev_order) {
        if (w->input.has_gpu && seen_non_gpu) {
            return;
        }
        seen_non_gpu = seen_non_gpu || !w->input.has_gpu;
    }

    // First position where the fresh ordering disagrees with the previous one
    // decides it: everything before that is unchanged anyway.
    for (size_t i = 0; i < active.size(); ++i) {
        if (active[i]->input.node_id == prev_order[i]->input.node_id) {
            continue;
        }
        const double incumbent = prev_order[i]->input.score;
        const double challenger = active[i]->input.score;
        const double threshold =
                std::max(incumbent, 1.0) * (1.0 + LAYOUT_ROLE_HYSTERESIS_PERCENT / 100.0);
        if (challenger > threshold) {
            fprintf(stderr,
                    "layout_planner: role reorder accepted at position %zu -- %s (score %.1f) "
                    "displaces %s (score %.1f), gain exceeds %.0f%% hysteresis\n",
                    i, active[i]->input.node_id.c_str(), challenger,
                    prev_order[i]->input.node_id.c_str(), incumbent,
                    LAYOUT_ROLE_HYSTERESIS_PERCENT);
            return;
        }
        fprintf(stderr,
                "layout_planner: keeping previous role order -- %s (score %.1f) does not beat "
                "%s (score %.1f) by the %.0f%% hysteresis margin\n",
                active[i]->input.node_id.c_str(), challenger,
                prev_order[i]->input.node_id.c_str(), incumbent,
                LAYOUT_ROLE_HYSTERESIS_PERCENT);
        active = prev_order;
        return;
    }
}

// Single-stream cost of an allocation: a token traverses every stage in
// sequence, so what matters is the SUM of stage times, not the max. Stage
// time is modelled as layers_i / decode_tps_i (Task 21.4; Petals/SWARM use
// the same measured-throughput input, arXiv:2312.08361, 2301.11913).
//
// Note this deliberately does NOT balance load: balancing minimizes the
// slowest stage, which is the right objective only when many requests are in
// flight at once. With one stream there is nothing to overlap.
static double single_stream_cost(
        const std::vector<int> & counts,
        const std::vector<layout_internal_node *> & active) {
    double total = 0.0;
    for (size_t i = 0; i < counts.size() && i < active.size(); ++i) {
        const double tps = active[i]->input.decode_tps;
        if (tps <= 0.0) {
            return -1.0;
        }
        total += static_cast<double>(counts[i]) / tps;
    }
    return total;
}

static bool all_nodes_have_measured_tps(const std::vector<layout_internal_node *> & active) {
    for (const auto * w : active) {
        if (w->input.decode_tps <= 0.0) {
            return false;
        }
    }
    return true;
}

// Task 21.4: allocate layers by measured throughput instead of by score
// proportion. Minimizing sum(layers_i / tps_i) has a degenerate optimum --
// put everything on the fastest node -- so memory is what actually shapes the
// result: fill the fastest node to its cap, then the next fastest, and so on.
// That the split exists at all is a memory constraint, not a speed strategy.
//
// `caps` bounds each node; the caller still runs its own clamp/redistribute
// pass afterwards, so this only has to produce a sane starting point.
static std::vector<int> counts_by_measured_throughput(
        const std::vector<layout_internal_node *> & active,
        const std::vector<int> & caps,
        int n_layer) {
    std::vector<size_t> by_speed(active.size());
    for (size_t i = 0; i < active.size(); ++i) {
        by_speed[i] = i;
    }
    std::sort(by_speed.begin(), by_speed.end(), [&active](size_t a, size_t b) {
        if (active[a]->input.decode_tps != active[b]->input.decode_tps) {
            return active[a]->input.decode_tps > active[b]->input.decode_tps;
        }
        return active[a]->input.node_id < active[b]->input.node_id;
    });

    std::vector<int> counts(active.size(), 0);
    int remaining = n_layer;
    for (const size_t idx : by_speed) {
        if (remaining <= 0) {
            break;
        }
        const int take = std::min(remaining, std::max(0, caps[idx]));
        counts[idx] = take;
        remaining -= take;
    }
    return counts;
}

// Layer-count hysteresis. Same reasoning as apply_role_hysteresis, but the
// thing being protected is more expensive: a changed count means weights
// actually move between nodes. Keeps the previous allocation unless the new
// one improves the single-stream cost by more than
// LAYOUT_COUNT_HYSTERESIS_PERCENT.
static void apply_count_hysteresis(
        std::vector<int> & counts,
        const std::vector<int> & caps,
        const std::vector<layout_internal_node *> & active,
        const desired_model_layout * previous,
        int n_layer) {
    if (previous == nullptr || previous->placements.empty()) {
        return;
    }
    if (!all_nodes_have_measured_tps(active)) {
        return;  // no cost model -> nothing to compare, leave counts alone
    }

    std::map<std::string, int> prev_counts;
    for (const auto & p : previous->placements) {
        prev_counts[p.node_id]++;
    }

    // The previous allocation has to still be usable today: same node set,
    // same total, and within each node's current memory cap.
    if (prev_counts.size() != active.size()) {
        return;
    }
    std::vector<int> prev(active.size(), 0);
    int prev_total = 0;
    for (size_t i = 0; i < active.size(); ++i) {
        const auto it = prev_counts.find(active[i]->input.node_id);
        if (it == prev_counts.end() || it->second > caps[i]) {
            return;
        }
        prev[i] = it->second;
        prev_total += it->second;
    }
    if (prev_total != n_layer) {
        return;  // model or cluster changed shape; not a like-for-like compare
    }
    if (prev == counts) {
        return;
    }

    const double cost_prev = single_stream_cost(prev, active);
    const double cost_new  = single_stream_cost(counts, active);
    if (cost_prev <= 0.0 || cost_new <= 0.0) {
        return;
    }

    const double gain_percent = (cost_prev - cost_new) * 100.0 / cost_prev;
    if (gain_percent > LAYOUT_COUNT_HYSTERESIS_PERCENT) {
        fprintf(stderr,
                "layout_planner: layer rebalance accepted -- predicted single-stream gain "
                "%.1f%% exceeds %.0f%% threshold\n",
                gain_percent, LAYOUT_COUNT_HYSTERESIS_PERCENT);
        return;
    }
    fprintf(stderr,
            "layout_planner: keeping previous layer counts -- predicted gain %.1f%% is under "
            "the %.0f%% threshold, not worth re-syncing weights\n",
            gain_percent, LAYOUT_COUNT_HYSTERESIS_PERCENT);
    counts = prev;
}

} // namespace

bool layout_has_no_overlap(const desired_model_layout & layout) {
    std::set<int32_t> seen;
    for (const auto & p : layout.placements) {
        if (p.layer_index < 0) {
            return false;
        }
        if (!seen.insert(p.layer_index).second) {
            return false;
        }
    }
    return true;
}

bool layout_has_full_coverage(const desired_model_layout & layout, int32_t n_layer) {
    if (n_layer <= 0) {
        return layout.placements.empty();
    }
    if (static_cast<int32_t>(layout.placements.size()) != n_layer) {
        return false;
    }
    std::vector<bool> seen(static_cast<size_t>(n_layer), false);
    for (const auto & p : layout.placements) {
        if (p.layer_index < 0 || p.layer_index >= n_layer) {
            return false;
        }
        if (seen[static_cast<size_t>(p.layer_index)]) {
            return false;
        }
        seen[static_cast<size_t>(p.layer_index)] = true;
    }
    for (bool s : seen) {
        if (!s) {
            return false;
        }
    }
    return true;
}

bool layouts_placement_equal(
        const desired_model_layout & a,
        const desired_model_layout & b) {
    if (a.placements.size() != b.placements.size()) {
        return false;
    }

    std::vector<std::pair<int32_t, std::string>> pa;
    std::vector<std::pair<int32_t, std::string>> pb;
    pa.reserve(a.placements.size());
    pb.reserve(b.placements.size());
    for (const auto & p : a.placements) {
        pa.emplace_back(p.layer_index, p.node_id);
    }
    for (const auto & p : b.placements) {
        pb.emplace_back(p.layer_index, p.node_id);
    }
    std::sort(pa.begin(), pa.end());
    std::sort(pb.begin(), pb.end());
    return pa == pb;
}

bool validate_desired_layout(
        const desired_model_layout & layout,
        const model_manifest & manifest,
        std::string & error) {
    error.clear();
    const int32_t n_layer = manifest.n_layer > 0
            ? static_cast<int32_t>(manifest.n_layer)
            : static_cast<int32_t>(manifest.layers.size());

    if (!layout_has_no_overlap(layout)) {
        error = "duplicate layer indices in placements";
        return false;
    }
    if (!layout_has_full_coverage(layout, n_layer)) {
        error = "layout does not cover all layers exactly once";
        return false;
    }

    uint64_t weight_sum = 0;
    for (const auto & p : layout.placements) {
        const uint64_t expected = manifest_layer_weight_bytes(manifest, p.layer_index);
        if (expected > 0 && p.size_bytes != expected) {
            error = "layer " + std::to_string(p.layer_index) + " size mismatch";
            return false;
        }
        weight_sum += p.size_bytes;
    }

    uint64_t manifest_weight = 0;
    for (const auto & ld : manifest.layers) {
        manifest_weight += ld.size_bytes;
    }
    if (manifest_weight > 0 && weight_sum != manifest_weight) {
        error = "total placement weight does not match manifest";
        return false;
    }
    return true;
}

layout_build_result build_desired_layout(
        const std::string & model_id,
        const model_manifest & manifest,
        const std::vector<layout_node_input> & nodes,
        int32_t n_ctx,
        const desired_model_layout * previous) {
    layout_build_result result{};
    result.layout.model_id = model_id;

    if (manifest.empty() || manifest.layers.empty()) {
        result.error = "manifest has no layer descriptors";
        return result;
    }
    if (nodes.empty()) {
        result.error = "no nodes available";
        result.layout.fits_cluster = false;
        result.layout.warnings.push_back("cluster has no registered nodes");
        return result;
    }

    const model_memory_requirements mem = memory_requirements_from_manifest(manifest, n_ctx);
    const int32_t n_layer = mem.n_layer;
    if (n_layer <= 0) {
        result.error = "manifest reports zero layers";
        return result;
    }

    result.layout.total_weight_bytes = mem.weights_bytes;
    result.layout.total_required_memory = mem.total_bytes();

    std::vector<layout_internal_node> work;
    work.reserve(nodes.size());
    uint64_t cluster_budget = 0;
    for (const auto & n : nodes) {
        layout_internal_node w{};
        w.input = n;
        w.device = layout_normalize_device(n);
        w.budget_bytes = n.has_gpu ? n.gpu_budget_bytes : n.cpu_budget_bytes;
        if (w.budget_bytes == 0) {
            continue;
        }
        cluster_budget += w.budget_bytes;
        work.push_back(std::move(w));
    }

    if (work.empty()) {
        result.error = "no node has available memory budget";
        result.layout.fits_cluster = false;
        return result;
    }

    if (cluster_budget < mem.total_bytes()) {
        result.layout.fits_cluster = false;
        result.layout.warnings.push_back("cluster memory budget is below model requirement");
        result.error = "model does not fit in cluster memory";
        return result;
    }

    std::sort(work.begin(), work.end(), node_sort_order);

    std::vector<layout_internal_node *> active;
    active.reserve(work.size());
    for (auto & w : work) {
        const uint64_t min_cost = layer_placement_cost(mem, 0);
        if (w.budget_bytes >= min_cost) {
            active.push_back(&w);
        }
    }

    if (active.empty()) {
        result.layout.fits_cluster = false;
        result.error = "no node can hold even one layer";
        return result;
    }

    apply_role_hysteresis(active, previous);

    // Per-node memory caps. Independent of how layers are allocated, so it is
    // computed first -- the measured-throughput allocation below needs to know
    // where each node runs out of room.
    std::vector<int> caps(active.size(), 0);
    int cap_cursor = 0;
    for (size_t i = 0; i < active.size(); ++i) {
        uint64_t used = 0;
        int local = cap_cursor;
        while (local < n_layer) {
            const uint64_t cost = layer_placement_cost(mem, local);
            if (cost == 0 || cost > active[i]->budget_bytes || used + cost > active[i]->budget_bytes) {
                break;
            }
            used += cost;
            ++local;
        }
        caps[i] = local - cap_cursor;
    }

    std::vector<int> counts;
    if (all_nodes_have_measured_tps(active)) {
        // Task 21.4: measured throughput drives the split.
        counts = counts_by_measured_throughput(active, caps, n_layer);
    } else {
        // Fallback whenever any node has not reported a benchmark yet: the
        // original score-proportional split. Score is always populated, so
        // this keeps a cold or partially-registered cluster working exactly
        // as before.
        std::vector<double> scores;
        scores.reserve(active.size());
        for (const auto * w : active) {
            scores.push_back(w->input.score > 0.0 ? w->input.score : 1.0);
        }

        double score_total = 0.0;
        for (double s : scores) {
            score_total += s;
        }
        if (score_total <= 0.0) {
            score_total = static_cast<double>(scores.size());
        }

        counts.assign(active.size(), 0);
        int assigned = 0;
        struct frac { size_t idx; double rem; };
        std::vector<frac> fracs;
        for (size_t i = 0; i < active.size(); ++i) {
            const double exact = static_cast<double>(n_layer) * scores[i] / score_total;
            counts[i] = static_cast<int>(std::floor(exact));
            assigned += counts[i];
            fracs.push_back({ i, exact - static_cast<double>(counts[i]) });
        }
        int leftover = n_layer - assigned;
        std::sort(fracs.begin(), fracs.end(), [](const frac & a, const frac & b) {
            if (a.rem != b.rem) {
                return a.rem > b.rem;
            }
            return a.idx < b.idx;
        });
        for (int i = 0; i < leftover && i < static_cast<int>(active.size()); ++i) {
            counts[fracs[static_cast<size_t>(i)].idx]++;
        }
    }

    if (n_layer >= static_cast<int>(active.size())) {
        for (size_t i = 0; i < active.size(); ++i) {
            if (counts[i] > 0) {
                continue;
            }
            size_t donor = 0;
            for (size_t j = 1; j < active.size(); ++j) {
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

    for (int iter = 0; iter < static_cast<int>(active.size()) + 1; ++iter) {
        int excess = 0;
        double slack_score = 0.0;
        for (size_t i = 0; i < active.size(); ++i) {
            if (counts[i] > caps[i]) {
                excess += counts[i] - caps[i];
                counts[i] = caps[i];
            }
            if (counts[i] < caps[i]) {
                slack_score += active[i]->input.score;
            }
        }
        if (excess == 0) {
            break;
        }
        int remaining = excess;
        for (size_t i = 0; i < active.size() && remaining > 0; ++i) {
            if (counts[i] >= caps[i] || slack_score <= 0.0) {
                continue;
            }
            const int share = std::max(1, static_cast<int>(std::round(
                    static_cast<double>(excess) * active[i]->input.score / slack_score)));
            const int room = caps[i] - counts[i];
            const int add = std::min({ share, remaining, room });
            counts[i] += add;
            remaining -= add;
        }
        for (size_t i = 0; i < active.size() && remaining > 0; ++i) {
            if (counts[i] >= caps[i]) {
                continue;
            }
            const int room = caps[i] - counts[i];
            const int add = std::min(room, remaining);
            counts[i] += add;
            remaining -= add;
        }
        if (remaining > 0) {
            result.layout.fits_cluster = false;
            result.error = "cannot fit model layers within node memory budgets";
            return result;
        }
    }

    // Only now, against a feasible allocation, decide whether the change is
    // worth physically moving weights for.
    apply_count_hysteresis(counts, caps, active, previous, n_layer);

    int total_assigned = 0;
    for (int c : counts) {
        total_assigned += c;
    }
    if (total_assigned != n_layer) {
        result.layout.fits_cluster = false;
        result.error = "layer assignment incomplete: " + std::to_string(total_assigned) +
                        "/" + std::to_string(n_layer);
        return result;
    }

    // Pipeline role order. The node holding the first layers becomes the
    // entry stage and the node holding the last layers becomes final. Final
    // is the heaviest per-token stage (output norm + lm_head + sampler on
    // top of its layers; see Research 17), so it goes to the strongest node
    // (active[0]); entry goes to the second-strongest, and the rest fill the
    // middle in score order.
    //
    // Deliberately score-only, no network input. An earlier version
    // (Task 21.1) picked entry by measured RTT to final, on the theory that
    // entry carries the client control connection and the fa-link draft
    // delivery. That was reverted 2026-07-24: RTT on a home network is
    // noisy and non-stationary, so the chosen entry node kept changing
    // between sessions, and every change orphaned that model's already-
    // installed layers on the previous node -- surfacing as recurring
    // DEGRADED coverage and "runtime coverage not ready" session failures
    // that had to be repaired by hand. The latency win was never measured
    // (the A/B in TASK_21_PROVEN_PRACTICES_PLAN.md Item 1 was never run),
    // while the layer-churn cost was real and repeated. Placement stability
    // beats an unquantified few-ms gain: score is stable across sessions,
    // RTT is not.
    std::vector<size_t> stage_order;
    stage_order.reserve(active.size());
    if (active.size() >= 2) {
        for (size_t i = 1; i < active.size(); ++i) {
            stage_order.push_back(i);
        }
        stage_order.push_back(0);
        fprintf(stderr, "layout_planner: entry=%s final=%s (score order)\n",
                active[1]->input.node_id.c_str(), active[0]->input.node_id.c_str());
    } else {
        stage_order.push_back(0);
    }

    int layer_cursor = 0;
    for (const size_t i : stage_order) {
        for (int c = 0; c < counts[i]; ++c) {
            layer_placement p{};
            p.layer_index = layer_cursor;
            p.node_id     = active[i]->input.node_id;
            p.device      = active[i]->device;
            p.size_bytes  = manifest_layer_weight_bytes(manifest, layer_cursor);
            if (p.size_bytes == 0 && layer_cursor < static_cast<int>(mem.layers.size())) {
                p.size_bytes = mem.layers[static_cast<size_t>(layer_cursor)].weight_bytes;
            }
            p.required = true;
            result.layout.placements.push_back(std::move(p));
            ++layer_cursor;
        }
    }

    if (layer_cursor != n_layer) {
        result.error = "internal layer cursor mismatch";
        result.layout.fits_cluster = false;
        return result;
    }

    std::string verr;
    if (!validate_desired_layout(result.layout, manifest, verr)) {
        result.error = verr;
        result.layout.fits_cluster = false;
        return result;
    }

    result.layout.fits_cluster = true;
    result.success = true;
    return result;
}
