// Relayout role hysteresis (Task 21.5): node score is measured live and
// drifts under unrelated load, so a marginal score change must not be able
// to move a model's layers between nodes. See the Task 21.1 revert in
// docs/KNOWN_ISSUES.md for what unchecked role churn costs.

#include "orchestrator/layout_planner/layout_planner.h"
#include "test_layout_common.h"

#include <algorithm>
#include <cstdio>
#include <set>
#include <string>
#include <vector>

// Pipeline order: entry, middle..., final. build_desired_layout gives final
// (heaviest stage) to the strongest node and entry to the second strongest,
// so for scores c > b > a the pipeline reads entry=b, middle=a, final=c.
static std::vector<std::string> role_order(const desired_model_layout & layout) {
    std::vector<const layer_placement *> sorted;
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

static std::string join(const std::vector<std::string> & v) {
    std::string s;
    for (const auto & x : v) {
        if (!s.empty()) {
            s += ",";
        }
        s += x;
    }
    return s;
}

static int fail(const char * what, const std::vector<std::string> & got,
        const std::vector<std::string> & want) {
    fprintf(stderr, "test-layout-hysteresis: %s -- got [%s], want [%s]\n",
            what, join(got).c_str(), join(want).c_str());
    return 1;
}

int main() {
    const model_manifest manifest = make_test_manifest(30, 100 * 1024 * 1024);

    // Baseline: c strongest, then b, then a. All CPU-only so the has_gpu
    // hard-ordering rule stays out of the way.
    const std::vector<layout_node_input> base = {
        make_layout_node("a", 100.0, 2.0, 0.0),
        make_layout_node("b", 105.0, 2.0, 0.0),
        make_layout_node("c", 111.0, 2.0, 0.0),
    };
    const auto first = build_desired_layout("m", manifest, base);
    if (!first.success || !first.layout.fits_cluster) {
        fprintf(stderr, "test-layout-hysteresis: baseline build failed: %s\n", first.error.c_str());
        return 1;
    }
    // scores c=111 > b=105 > a=100  ->  entry=b, middle=a, final=c
    const std::vector<std::string> baseline_order = role_order(first.layout);
    const std::vector<std::string> want_baseline = { "b", "a", "c" };
    if (baseline_order != want_baseline) {
        return fail("baseline order", baseline_order, want_baseline);
    }

    // 1. Marginal drift: a jumps 100 -> 114, enough to top the raw sort but
    //    only +2.7% over c's 111 -- under the 10% margin, so the previous
    //    order must be kept and no layer moves.
    {
        std::vector<layout_node_input> drifted = base;
        drifted[0].score = 114.0;
        const auto built = build_desired_layout(
                "m", manifest, drifted, 0, &first.layout);
        if (!built.success) {
            fprintf(stderr, "test-layout-hysteresis: marginal build failed: %s\n",
                    built.error.c_str());
            return 1;
        }
        const auto got = role_order(built.layout);
        if (got != want_baseline) {
            return fail("marginal drift should not reorder", got, want_baseline);
        }
    }

    // 2. Decisive gain: a jumps to 200, far past c's 111 + 10%. A real
    //    capability change should still be honored.
    {
        std::vector<layout_node_input> improved = base;
        improved[0].score = 200.0;
        const auto built = build_desired_layout(
                "m", manifest, improved, 0, &first.layout);
        if (!built.success) {
            fprintf(stderr, "test-layout-hysteresis: decisive build failed: %s\n",
                    built.error.c_str());
            return 1;
        }
        // a=200 > c=111 > b=105  ->  final=a, entry=c, middle=b
        const auto got = role_order(built.layout);
        const std::vector<std::string> want = { "c", "b", "a" };
        if (got != want) {
            return fail("decisive gain must reorder", got, want);
        }
    }

    // 3. No previous layout (nullptr): pure score order, hysteresis inactive.
    {
        std::vector<layout_node_input> drifted = base;
        drifted[0].score = 114.0;
        // a=114 > c=111 > b=105  ->  final=a, entry=c, middle=b
        const auto built = build_desired_layout("m", manifest, drifted);
        const auto got = role_order(built.layout);
        const std::vector<std::string> want = { "c", "b", "a" };
        if (got != want) {
            return fail("nullptr previous must use raw score order", got, want);
        }
    }

    // 4. Topology change (node dropped) is not score noise -- the surviving
    //    nodes take the fresh ordering rather than an order derived from a
    //    layout that mentions a node which is no longer there.
    {
        const std::vector<layout_node_input> two = {
            make_layout_node("a", 300.0, 2.0, 0.0),
            make_layout_node("b", 105.0, 2.0, 0.0),
        };
        const auto built = build_desired_layout("m", manifest, two, 0, &first.layout);
        if (!built.success) {
            fprintf(stderr, "test-layout-hysteresis: node-loss build failed: %s\n",
                    built.error.c_str());
            return 1;
        }
        // a=300 > b=105  ->  final=a, entry=b
        const auto got = role_order(built.layout);
        const std::vector<std::string> want = { "b", "a" };
        if (got != want) {
            return fail("node loss must use fresh order", got, want);
        }
    }

    // 5. Repeated marginal jitter must be a fixed point: feeding the kept
    //    layout back in, with scores wobbling either way, never moves layers.
    //    This is the property the whole mechanism exists for.
    {
        desired_model_layout carried = first.layout;
        const double wobble[] = { 114.0, 96.0, 113.0, 108.0, 114.0 };
        for (const double s : wobble) {
            std::vector<layout_node_input> drifted = base;
            drifted[0].score = s;
            const auto built = build_desired_layout(
                    "m", manifest, drifted, 0, &carried);
            if (!built.success) {
                fprintf(stderr, "test-layout-hysteresis: jitter build failed at score %.0f: %s\n",
                        s, built.error.c_str());
                return 1;
            }
            const auto got = role_order(built.layout);
            if (got != want_baseline) {
                return fail("jitter must be a fixed point", got, want_baseline);
            }
            carried = built.layout;
        }
    }

    printf("test-layout-hysteresis: OK (margin %.0f%%: marginal drift held, "
           "decisive gain honored, jitter is a fixed point)\n",
            LAYOUT_ROLE_HYSTERESIS_PERCENT);
    return 0;
}
