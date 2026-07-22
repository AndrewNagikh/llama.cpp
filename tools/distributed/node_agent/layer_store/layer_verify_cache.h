#pragma once

#include <cstdint>
#include <string>

// Caches the result of a full-content checksum verification
// (layer_store::verify_layer / verify_blob_tensor) so that repeated,
// frequent "is this model's coverage still ready" polls -- see
// node_agent.cpp's /installed-layers handler, hit on every orchestrator
// /session/create -- don't re-read and re-hash the model's full weight
// data off disk on every single call. Real corruption detection still
// happens on write (node_agent/synchronization/executors/http_range/
// http_range_executor.cpp already verifies right after storing a blob,
// and seeds this cache with that result) and via the dedicated
// orchestrator/consistency/store_verifier sweep; this cache only bounds
// how often the coverage-poll hot path repeats that work.
// See docs/bench/2026-07-23_g3_soak/G3_SOAK_REPORT.md for the measured
// cost of not having this (qwen2.5-32b's coverage poll alone was slow
// enough to fail ~50% of session-creates under soak).

// Returns true and fills `out_ready` if a cached result exists and is
// still within `ttl_sec` of when it was recorded. Returns false (cache
// miss or stale) if the caller must verify for real.
bool layer_verify_cache_get(
        const std::string & model_id,
        const std::string & blob_key,
        int64_t ttl_sec,
        bool & out_ready);

// Records a freshly-computed verification result, timestamped now.
void layer_verify_cache_put(
        const std::string & model_id,
        const std::string & blob_key,
        bool ready);
