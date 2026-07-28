// Task 24, principle 2: offline is not lost.
//
// A layer sitting on a node that is switched off is still on that node's
// disk. Until 2026-07-28 compute_coverage() put those layers into
// `missing` and reported DEGRADED, so a closed laptop was indistinguishable
// from data loss -- which is what made healthy models look broken and
// prompted manual "repairs" of models that were never damaged.
//
// These cases pin that distinction down.

#include "orchestrator/coverage/coverage.h"

#include <cstdio>
#include <string>

static int g_failures = 0;

static void check(const char * name, bool cond, const std::string & detail) {
    if (cond) {
        printf("  ok   %-52s %s\n", name, detail.c_str());
        return;
    }
    printf("  FAIL %-52s %s\n", name, detail.c_str());
    ++g_failures;
}

// Three layers each on node-a, node-b, node-c.
static desired_model_layout layout_abc() {
    desired_model_layout d;
    d.model_id = "test-model";
    const char * nodes[] = { "node-a", "node-b", "node-c" };
    for (int i = 0; i < 9; ++i) {
        layer_placement p;
        p.layer_index = i;
        p.node_id     = nodes[i / 3];
        d.placements.push_back(p);
    }
    return d;
}

// Everything installed and ready, exactly where the layout wants it.
static actual_model_layout all_installed() {
    actual_model_layout a;
    a.model_id = "test-model";
    const char * nodes[] = { "node-a", "node-b", "node-c" };
    for (int i = 0; i < 9; ++i) {
        installed_layer l;
        l.layer_index = i;
        l.node_id     = nodes[i / 3];
        l.state       = install_state::ready;
        a.layers.push_back(l);
    }
    return a;
}

static std::string summary(const coverage_report & r) {
    return "state=" + coverage_state_to_string(r.state) +
           " ready=" + std::to_string(r.ready_layers) +
           " missing=" + std::to_string(r.missing_layers) +
           " unavailable=" + std::to_string(r.unavailable_layers);
}

int main() {
    const desired_model_layout desired = layout_abc();
    const actual_model_layout  actual  = all_installed();

    {   // Baseline: everyone up, everything present.
        const coverage_report r = compute_coverage(desired, actual, { "node-a", "node-b", "node-c" });
        check("all nodes online -> READY", r.state == coverage_state::ready, summary(r));
        check("  nothing counted unavailable", r.unavailable_layers == 0, summary(r));
    }

    {   // The case this file exists for. node-a is off; its 3 layers are on
        // its disk and must not be called missing, because every consumer
        // reads `missing` as "download this".
        const coverage_report r = compute_coverage(desired, actual, { "node-b", "node-c" });
        check("node-a offline -> UNAVAILABLE, not DEGRADED",
                r.state == coverage_state::unavailable, summary(r));
        check("  its layers are unavailable, not missing",
                r.unavailable_layers == 3 && r.missing_layers == 0, summary(r));
        check("  visible layers still counted ready", r.ready_layers == 6, summary(r));
        // The name is what makes the message actionable: "start node-a", not
        // "a node is down".
        check("  names the machine to switch on",
                r.unavailable_nodes.size() == 1 && r.unavailable_nodes[0] == "node-a",
                r.unavailable_nodes.empty() ? "(none)" : r.unavailable_nodes[0]);

        // The install planner consumes this. If an absent node's layers leaked
        // into `missing` here, returning from a reboot would re-download them.
        const reconciliation_result rec = reconcile_layers(desired, actual, { "node-b", "node-c" });
        check("  install plan has nothing to fetch", rec.missing.empty(),
                "missing=" + std::to_string(rec.missing.size()));
    }

    {   // Genuine absence on a node we can see is still missing, and still
        // outranks unavailability -- a real gap must not hide behind an
        // offline node.
        actual_model_layout damaged = actual;
        damaged.layers.erase(damaged.layers.begin() + 4);   // node-b, layer 4
        const coverage_report r = compute_coverage(desired, damaged, { "node-b", "node-c" });
        check("missing on an online node still reported",
                r.missing_layers == 1, summary(r));
        check("  and outranks the offline node",
                r.state == coverage_state::partial, summary(r));
    }

    {   // Damage outranks everything: it is the only case needing real repair.
        actual_model_layout corrupt = actual;
        corrupt.layers[4].state = install_state::corrupted;
        const coverage_report r = compute_coverage(desired, corrupt, { "node-b", "node-c" });
        check("corruption outranks an offline node",
                r.state == coverage_state::degraded, summary(r));
    }

    {   // Round-trip, so a persisted registry keeps the distinction instead of
        // silently decaying back to DEGRADED on restart.
        const coverage_report r = compute_coverage(desired, actual, { "node-b", "node-c" });
        const coverage_report back = coverage_report::from_json(r.to_json());
        check("survives JSON round-trip",
                back.state == coverage_state::unavailable &&
                back.unavailable_layers == 3 &&
                back.missing_layers == 0, summary(back));
    }

    if (g_failures == 0) {
        printf("test-coverage-offline: all checks passed\n");
        return 0;
    }
    printf("test-coverage-offline: %d check(s) failed\n", g_failures);
    return 1;
}
