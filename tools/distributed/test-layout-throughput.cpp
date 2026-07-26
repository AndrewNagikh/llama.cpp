// Measured-throughput layer partitioning (Task 21.4) + layer-count hysteresis
// (Task 21.5). Layers follow measured decode_tps, bounded by memory -- and a
// count change only happens when it is worth re-syncing weights for.

#include "orchestrator/layout_planner/layout_planner.h"
#include "test_layout_common.h"

#include <cstdio>
#include <map>
#include <string>
#include <vector>

static std::map<std::string, int> counts_of(const desired_model_layout & layout) {
    std::map<std::string, int> counts;
    for (const auto & p : layout.placements) {
        counts[p.node_id]++;
    }
    return counts;
}

static std::string describe(const std::map<std::string, int> & counts) {
    std::string s;
    for (const auto & kv : counts) {
        if (!s.empty()) {
            s += " ";
        }
        s += kv.first + "=" + std::to_string(kv.second);
    }
    return s;
}

static int fail(const char * what, const std::map<std::string, int> & got) {
    fprintf(stderr, "test-layout-throughput: %s -- got [%s]\n", what, describe(got).c_str());
    return 1;
}

static layout_node_input node_with_tps(
        const std::string & id, double score, double ram_gb, double tps) {
    layout_node_input n = make_layout_node(id, score, ram_gb, 0.0);
    n.decode_tps = tps;
    return n;
}

int main() {
    // 30 layers x 100MB = 3GB of weights.
    const model_manifest manifest = make_test_manifest(30, 100 * 1024 * 1024);

    // Every node has room for the whole model, so memory never binds here and
    // throughput alone decides. c is 10x faster than a.
    const std::vector<layout_node_input> roomy = {
        node_with_tps("a", 100.0, 8.0, 5.0),
        node_with_tps("b", 100.0, 8.0, 10.0),
        node_with_tps("c", 100.0, 8.0, 50.0),
    };

    // 1. With memory to spare, the fastest node should take essentially
    //    everything -- minimizing sum(layers/tps) has no reason to spread.
    //    (The planner still guarantees every active node at least one layer,
    //    so a and b keep 1 each.)
    desired_model_layout concentrated;
    {
        const auto built = build_desired_layout("m", manifest, roomy);
        if (!built.success || !built.layout.fits_cluster) {
            fprintf(stderr, "test-layout-throughput: roomy build failed: %s\n",
                    built.error.c_str());
            return 1;
        }
        const auto got = counts_of(built.layout);
        if (got.at("c") < 28) {
            return fail("fastest node should hold nearly all layers", got);
        }
        if (got.at("a") + got.at("b") + got.at("c") != 30) {
            return fail("layers must sum to 30", got);
        }
        concentrated = built.layout;
    }

    // 2. Same nodes, but the fast one can only hold ~10 layers. Memory is what
    //    forces the split; the remainder must go to the next-fastest node
    //    first, not be spread evenly.
    {
        std::vector<layout_node_input> capped = roomy;
        capped[2] = node_with_tps("c", 100.0, 1.05, 50.0);  // ~10 layers
        capped[1] = node_with_tps("b", 100.0, 8.0, 10.0);
        capped[0] = node_with_tps("a", 100.0, 8.0, 5.0);

        const auto built = build_desired_layout("m", manifest, capped);
        if (!built.success || !built.layout.fits_cluster) {
            fprintf(stderr, "test-layout-throughput: capped build failed: %s\n",
                    built.error.c_str());
            return 1;
        }
        const auto got = counts_of(built.layout);
        if (got.at("c") > 11) {
            return fail("c must be bounded by its memory cap", got);
        }
        if (got.at("b") <= got.at("a")) {
            return fail("spillover should prefer the faster remaining node (b > a)", got);
        }
    }

    // 3. No decode_tps anywhere -> score-proportional fallback, i.e. equal
    //    scores give a roughly even split rather than a concentrated one.
    {
        const std::vector<layout_node_input> unmeasured = {
            make_layout_node("a", 100.0, 8.0, 0.0),
            make_layout_node("b", 100.0, 8.0, 0.0),
            make_layout_node("c", 100.0, 8.0, 0.0),
        };
        const auto built = build_desired_layout("m", manifest, unmeasured);
        const auto got = counts_of(built.layout);
        if (got.at("a") != 10 || got.at("b") != 10 || got.at("c") != 10) {
            return fail("unmeasured cluster must fall back to even score split", got);
        }
    }

    // 4. Partially measured (one node never benchmarked) also falls back --
    //    mixing a measured node against an assumed one would be worse than
    //    using the metric that exists for all of them.
    {
        std::vector<layout_node_input> partial = roomy;
        partial[1].decode_tps = 0.0;
        const auto built = build_desired_layout("m", manifest, partial);
        const auto got = counts_of(built.layout);
        if (got.at("a") != 10 || got.at("b") != 10 || got.at("c") != 10) {
            return fail("partially measured cluster must fall back", got);
        }
    }

    // 5. The case count hysteresis actually exists for: two near-identical
    //    nodes swap speed ranking. Greedy fill keys off the ORDER of tps, not
    //    its magnitude, so a 1% flip between a and b would otherwise migrate
    //    ~18 layers between them for a fraction of a percent of real gain.
    //
    //    Memory caps are set so a and b hold most of the model, making the
    //    swap expensive and the test meaningful.
    const std::vector<layout_node_input> contested = {
        node_with_tps("a", 100.0, 2.1, 9.9),    // ~20 layers of room
        node_with_tps("b", 100.0, 2.1, 10.0),   // marginally faster
        node_with_tps("c", 100.0, 1.05, 50.0),  // fastest, but only ~10 layers
    };

    desired_model_layout contested_layout;
    {
        const auto built = build_desired_layout("m", manifest, contested);
        if (!built.success || !built.layout.fits_cluster) {
            fprintf(stderr, "test-layout-throughput: contested build failed: %s\n",
                    built.error.c_str());
            return 1;
        }
        const auto got = counts_of(built.layout);
        if (got.at("b") <= got.at("a")) {
            return fail("b is faster, so b should hold the bulk", got);
        }
        contested_layout = built.layout;
    }
    {
        std::vector<layout_node_input> flipped = contested;
        flipped[0].decode_tps = 10.1;  // a now edges ahead of b by 1%
        const auto built = build_desired_layout(
                "m", manifest, flipped, 0, &contested_layout);
        if (!built.success) {
            fprintf(stderr, "test-layout-throughput: flip build failed: %s\n",
                    built.error.c_str());
            return 1;
        }
        const auto got = counts_of(built.layout);
        const auto want = counts_of(contested_layout);
        if (got != want) {
            fprintf(stderr, "test-layout-throughput: a 1%% ranking flip must not migrate "
                    "layers -- got [%s], want [%s]\n",
                    describe(got).c_str(), describe(want).c_str());
            return 1;
        }
    }

    // 6. A real capability change still gets through: c collapses to 2 tok/s,
    //    slower than both peers, so keeping the layers there is indefensible.
    {
        std::vector<layout_node_input> collapsed = roomy;
        collapsed[2].decode_tps = 2.0;
        const auto built = build_desired_layout(
                "m", manifest, collapsed, 0, &concentrated);
        if (!built.success) {
            fprintf(stderr, "test-layout-throughput: collapse build failed: %s\n",
                    built.error.c_str());
            return 1;
        }
        const auto got = counts_of(built.layout);
        const auto before = counts_of(concentrated);
        if (got.at("c") >= before.at("c")) {
            return fail("a collapsed node must lose its layers", got);
        }
    }

    // 7. Repeated ranking jitter between the two contested nodes is a fixed
    //    point: feed the layout back in while a and b trade the lead, and no
    //    layer ever moves. This is the property the mechanism exists for.
    {
        desired_model_layout carried = contested_layout;
        const double wobble[] = { 10.1, 9.8, 10.2, 9.95, 10.05 };
        const auto want = counts_of(contested_layout);
        for (const double tps : wobble) {
            std::vector<layout_node_input> drifted = contested;
            drifted[0].decode_tps = tps;
            const auto built = build_desired_layout("m", manifest, drifted, 0, &carried);
            if (!built.success) {
                fprintf(stderr, "test-layout-throughput: jitter build failed at %.2f: %s\n",
                        tps, built.error.c_str());
                return 1;
            }
            const auto got = counts_of(built.layout);
            if (got != want) {
                fprintf(stderr, "test-layout-throughput: jitter must be a fixed point at "
                        "tps=%.2f -- got [%s], want [%s]\n",
                        tps, describe(got).c_str(), describe(want).c_str());
                return 1;
            }
            carried = built.layout;
        }
    }

    printf("test-layout-throughput: OK (measured split, memory-bounded spillover, "
           "fallback when unmeasured, %.0f%% count hysteresis holds)\n",
            LAYOUT_COUNT_HYSTERESIS_PERCENT);
    return 0;
}
