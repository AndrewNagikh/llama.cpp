#pragma once

#include "coverage/coverage.h"

#include <set>
#include <string>

// ---------------------------------------------------------------------------
// Layout policy (Task 24)
//
// Decides what may be done to a model's layout. Split out as a pure function
// because the wrong answer here deletes data: on 2026-07-24 the orchestrator
// recomputed layouts from a view of the cluster that had not been verified,
// moved layers onto whichever node happened to be up, and deleted the copies
// on the others.
//
// The governing inversion: installed layers are the durable, expensive state
// and the layout is a cheap derived artifact. Historically it was backwards --
// the layout was recomputed freely and blobs on disk were expected to follow.
// See docs/TASK_24_CLUSTER_STABILITY.md.
// ---------------------------------------------------------------------------

enum class layout_action {
    // Stored layout stands. Nothing to do.
    keep,

    // Layers are missing or misplaced relative to the stored layout. Fix the
    // data, never the plan: an install plan built against the STORED layout
    // moves blobs toward it and leaves placement untouched.
    repair,

    // Build a fresh layout. Deliberately rare -- only when there is no layout
    // to protect, or the stored one cannot produce a runnable pipeline.
    recompute,

    // A node this layout needs is not online. Its layers are still on its
    // disk; the model is simply unavailable until it returns. Reassigning its
    // share is how a temporary absence becomes permanent loss.
    unavailable,
};

const char * layout_action_name(layout_action a);

struct layout_policy_input {
    // A stored layout exists and has placements.
    bool has_stored_layout = false;

    // Nodes the stored layout places layers on.
    std::set<std::string> layout_nodes;

    // Nodes online right now.
    std::set<std::string> online_nodes;

    // Whether the coverage below was polled from the cluster just now, as
    // opposed to loaded from the registry or left over from an earlier
    // moment. Stale coverage must never justify a destructive action -- this
    // is the flag whose absence caused the 2026-07-24 data loss.
    bool coverage_fresh = false;

    coverage_state coverage = coverage_state::empty;
    int missing_layers = 0;
    int corrupted_layers = 0;
    int total_layers = 0;
    int ready_layers = 0;
};

// Pure decision, no I/O. `reason` receives a short human-readable
// justification suitable for a log line.
layout_action decide_layout_action(const layout_policy_input & in, std::string & reason);
