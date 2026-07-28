#include "layout_policy.h"

const char * layout_action_name(layout_action a) {
    switch (a) {
        case layout_action::keep:        return "keep";
        case layout_action::repair:      return "repair";
        case layout_action::recompute:   return "recompute";
        case layout_action::unavailable: return "unavailable";
    }
    return "unknown";
}

layout_action decide_layout_action(const layout_policy_input & in, std::string & reason) {
    // Nothing installed yet, so there is no state to protect. This is the
    // ordinary first-install path and the main legitimate use of recompute.
    if (!in.has_stored_layout) {
        reason = "no stored layout yet";
        return layout_action::recompute;
    }

    // Order matters: this is checked before coverage is even consulted.
    // Coverage computed while a node is down reports that node's layers as
    // missing, which is true of the poll and false of the disk. Acting on it
    // is what turned a closed laptop into a re-download.
    for (const auto & node : in.layout_nodes) {
        if (in.online_nodes.count(node) == 0) {
            reason = "layout needs " + node + ", which is not online";
            return layout_action::unavailable;
        }
    }

    // Unverified state justifies nothing. Keeping is always safe; the next
    // pass with a fresh poll can still decide to repair.
    if (!in.coverage_fresh) {
        reason = "coverage not freshly polled";
        return layout_action::keep;
    }

    // A layout that cannot yield at least two pipeline stages is not a layout
    // this runtime can run -- the role loop never assigns `final` to a single
    // stage, which crashed the orchestrator outright on 2026-07-24. Recompute
    // it, but only when there are actually enough nodes to do better.
    if (in.layout_nodes.size() < 2 && in.online_nodes.size() >= 2) {
        reason = "stored layout uses " + std::to_string(in.layout_nodes.size()) +
                 " node(s) while " + std::to_string(in.online_nodes.size()) + " are online";
        return layout_action::recompute;
    }

    if (in.coverage == coverage_state::ready) {
        reason = "coverage ready";
        return layout_action::keep;
    }

    // Everything below here is a data problem, and every one of them is fixed
    // by moving blobs toward the stored layout -- never by choosing a new one.
    // Recomputing would relocate the very layers that are already correct and
    // orphan them, which is the loop that kept models permanently broken.
    if (in.corrupted_layers > 0) {
        reason = std::to_string(in.corrupted_layers) + " corrupted layer(s)";
        return layout_action::repair;
    }
    if (in.missing_layers > 0) {
        reason = std::to_string(in.missing_layers) + " missing layer(s)";
        return layout_action::repair;
    }

    // Every layer present and intact, yet not READY: blobs left on nodes an
    // earlier layout used. Cleanable, and emphatically not a reason to move
    // anything that is already in the right place.
    if (in.total_layers > 0 && in.ready_layers == in.total_layers) {
        reason = "all layers present but not ready -- stale blobs from an earlier layout";
        return layout_action::repair;
    }

    reason = "coverage " + coverage_state_to_string(in.coverage) + " with no identified cause";
    return layout_action::repair;
}
