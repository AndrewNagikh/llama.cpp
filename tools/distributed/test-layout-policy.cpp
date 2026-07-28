// Task 24 layout policy. Each case below is a situation that actually
// occurred on the cluster on 2026-07-24, and the expectation encodes what
// should have happened instead of what did.

#include "orchestrator/layout_policy/layout_policy.h"

#include <cstdio>
#include <string>

static int g_failures = 0;

static void expect(const char * name, layout_action want, const layout_policy_input & in) {
    std::string reason;
    const layout_action got = decide_layout_action(in, reason);
    if (got == want) {
        printf("  ok   %-46s -> %-11s (%s)\n", name, layout_action_name(got), reason.c_str());
        return;
    }
    printf("  FAIL %-46s -> %s, want %s (%s)\n",
            name, layout_action_name(got), layout_action_name(want), reason.c_str());
    ++g_failures;
}

// Healthy three-node model, coverage just polled.
static layout_policy_input healthy() {
    layout_policy_input in;
    in.has_stored_layout = true;
    in.layout_nodes      = { "node-a", "node-b", "node-c" };
    in.online_nodes      = { "node-a", "node-b", "node-c" };
    in.coverage_fresh    = true;
    in.coverage          = coverage_state::ready;
    in.total_layers      = 28;
    in.ready_layers      = 28;
    return in;
}

int main() {
    expect("healthy cluster", layout_action::keep, healthy());

    {   // First install: nothing on disk to protect.
        layout_policy_input in = healthy();
        in.has_stored_layout = false;
        expect("no stored layout", layout_action::recompute, in);
    }

    {   // The 2026-07-24 data loss. node-a was down; its layers were still on
        // its disk. Reassigning them is what made the absence permanent.
        layout_policy_input in = healthy();
        in.online_nodes = { "node-b", "node-c" };
        in.coverage     = coverage_state::degraded;
        in.ready_layers = 18;
        in.missing_layers = 10;
        expect("node offline", layout_action::unavailable, in);
    }

    {   // Offline takes priority over everything, including a coverage report
        // that looks catastrophic *because* the node is offline.
        layout_policy_input in = healthy();
        in.online_nodes   = { "node-b", "node-c" };
        in.coverage       = coverage_state::empty;
        in.ready_layers   = 0;
        in.missing_layers = 28;
        expect("node offline, coverage looks empty", layout_action::unavailable, in);
    }

    {   // Startup race: the first node to re-register triggered a sweep while
        // the others were still booting. Stale coverage justifies nothing.
        layout_policy_input in = healthy();
        in.coverage_fresh = false;
        in.coverage       = coverage_state::degraded;
        in.ready_layers   = 0;
        in.missing_layers = 28;
        expect("stale coverage", layout_action::keep, in);
    }

    {   // Single-node layout produces a one-stage pipeline, which has no final
        // role and crashed the orchestrator process.
        layout_policy_input in = healthy();
        in.layout_nodes = { "node-c" };
        expect("layout collapsed onto one node", layout_action::recompute, in);
    }

    {   // ...but not when there is genuinely only one node to run on. Then a
        // one-node layout is the only option and recomputing changes nothing.
        layout_policy_input in = healthy();
        in.layout_nodes = { "node-c" };
        in.online_nodes = { "node-c" };
        expect("one node total", layout_action::keep, in);
    }

    {   // gemma-3-1b: every layer present and intact, still not READY, because
        // of blobs left by an earlier layout. Recomputing here is what kept it
        // permanently broken -- each pass orphaned a fresh set.
        layout_policy_input in = healthy();
        in.coverage       = coverage_state::degraded;
        in.missing_layers = 0;
        in.corrupted_layers = 0;
        expect("all layers present, stale blobs", layout_action::repair, in);
    }

    {   // Real gap with the cluster whole: move data toward the stored layout,
        // never replace the layout.
        layout_policy_input in = healthy();
        in.coverage       = coverage_state::partial;
        in.ready_layers   = 18;
        in.missing_layers = 10;
        expect("layers genuinely missing", layout_action::repair, in);
    }

    {
        layout_policy_input in = healthy();
        in.coverage         = coverage_state::degraded;
        in.corrupted_layers = 2;
        expect("corrupted layers", layout_action::repair, in);
    }

    {   // A four-node cluster must behave identically -- nothing keys off the
        // number three.
        layout_policy_input in;
        in.has_stored_layout = true;
        in.layout_nodes      = { "node-a", "node-b", "node-c", "garage-pc" };
        in.online_nodes      = { "node-a", "node-b", "node-c", "garage-pc" };
        in.coverage_fresh    = true;
        in.coverage          = coverage_state::ready;
        in.total_layers      = 40;
        in.ready_layers      = 40;
        expect("four nodes, healthy", layout_action::keep, in);

        in.online_nodes.erase("garage-pc");
        in.coverage = coverage_state::degraded;
        expect("four nodes, one offline", layout_action::unavailable, in);
    }

    if (g_failures > 0) {
        printf("\ntest-layout-policy: %d FAILURE(S)\n", g_failures);
        return 1;
    }
    printf("\ntest-layout-policy: OK (layout is never recomputed to work around "
           "an offline node or unverified coverage)\n");
    return 0;
}
