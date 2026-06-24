# Upstream Compatibility Strategy

This document catalogs fork-specific changes and how we minimize merge conflict surface when pulling from [ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp).

**Fork point (approximate):** `bddfd2b11` (pre-distributed commits)  
**Branch:** `feature/distributed-runtime`

---

## Core Patch (minimal llama.cpp changes)

These files modify upstream `src/` or `common/` and are **required** for distributed inference. They should stay as small, isolated diffs.

| File | Purpose | Required? | Can move out? |
|------|---------|-----------|---------------|
| `src/llama-cparams.h` | `layer_start` / `layer_end` in context params | **Yes** | No — core state |
| `src/llama-context.h` | Hidden state buffer, `set_layer_range()` methods | **Yes** | No |
| `src/llama-context.cpp` | Layer-range validation, hidden injection in decode path, C API wrappers | **Yes** | No |
| `src/llama-graph.h` | `build_inp_hidden()` declaration, graph param compare | **Yes** | No |
| `src/llama-graph.cpp` | `build_inp_hidden()` — tensor input for mid-pipeline hidden state | **Yes** | No |
| `src/models/llama.cpp` | Partial layer loop `[layer_start, layer_end)`, `build_inp_hidden()` entry | **Yes** | No |
| `include/llama-distributed.h` | **Stable public API** for distributed runtime | **Yes** | No — this is the boundary |
| `common/common.h` | CLI `layer_start` / `layer_end` params (dev convenience) | Optional | Yes — could move to distributed-only tool |
| `common/common.cpp` | Calls `llama_set_layer_range()` from CLI params | Optional | Yes |
| `common/arg.cpp` | `--layer-start` / `--layer-end` flags | Optional | Yes |
| `tools/server/server-context.cpp` | Unrelated upstream delta on fork base | Review on merge | Maybe revert |
| `tools/server/README.md` | Doc delta | No | Yes |

**Stable API (use this, not `src/llama-ext.h`):**

```c
#include "llama-distributed.h"

llama_set_layer_range(ctx, start, end);
llama_set_hidden_state(ctx, data, n_tokens);
llama_clear_hidden_state(ctx);
```

Implementation remains in `src/llama-context.cpp`. Declarations live in `include/llama-distributed.h` only.

---

## Distributed Runtime (outside core)

All orchestration, networking, benchmarking, and workers live under **`tools/distributed/`**. None of this should be merged into `src/` or `common/`.

| Component | Location | Depends on core via |
|-----------|----------|---------------------|
| Orchestrator | `tools/distributed/orchestrator.cpp` | `llama-distributed.h`, `llama.h` |
| Node agent | `tools/distributed/node_agent.cpp` | same |
| Layer planner | `tools/distributed/layer_planner.*` | none (pure C++) |
| Node benchmark | `tools/distributed/node_benchmark.*` | `llama.h` |
| Node core (in-process alt) | `tools/distributed/node_core.*` | `llama-distributed.h` |
| Shared dist types | `tools/distributed/dist_common.*` | OS probes only |
| TCP transport | `tools/distributed/transport/split_tcp_wire.*` | sockets only |
| Pipeline workers | `tools/distributed/workers/split_gen3_*.cpp` | `llama-distributed.h` |
| Worker helpers | `tools/distributed/workers/split_gen_common.h` | `llama-distributed.h` |
| Integration tests | `tools/distributed/test-orchestrator-*.cpp` | HTTP + workers |
| Docs | `docs/task*.md`, `docs/architecture.md` | — |

**Legacy / proof-of-concept** (still under `tests/`, gated by `LLAMA_DISTRIBUTED`):

- `tests/split_sender.cpp`, `split_receiver.cpp` — 2-node TCP prototype
- `tests/split_gen_a.cpp`, `split_gen_b.cpp` — 2-node autoregressive
- `tests/test-partial-forward.cpp`, `test-injection-proof.cpp` — core API tests
- `tests/test-autoregressive-split*.cpp` — pipeline integration tests

---

## Patch Surface (metrics)

Measured against fork base `bddfd2b11..HEAD`:

| Metric | Before Task 6.9 | After Task 6.9 |
|--------|-----------------|----------------|
| Modified upstream files (`src/`, `common/`) | **10** | **10** (unchanged) |
| Lines changed in core (`src/` + `common/`) | ~294 insertions | ~294 + `llama-distributed.h` |
| Distributed-only files in `src/` | 0 | **0** |
| Distributed files under `tools/distributed/` | ~15 | **~20** (transport + workers subdirs) |
| Distributed files still in `tests/` | ~15 | **~10** (legacy POC tests only) |
| Public API header | `src/llama-ext.h` (internal) | **`include/llama-distributed.h`** |
| Build flag | always on | **`LLAMA_DISTRIBUTED=ON/OFF`** |
| Merge conflict risk (estimate) | High (unclear boundary) | **Medium** — core patch is 6 `src/` files |

**Total fork diff:** ~57 files, ~8450 lines added (mostly distributed runtime + tests).

---

## Design rules

1. **No distributed logic in `src/`** except partial-forward + hidden-state hooks.
2. **No `#include "../src/llama-ext.h"`** in orchestrator, node_agent, or workers.
3. **New distributed features** go under `tools/distributed/` only.
4. **Core API changes** go through `include/llama-distributed.h` with a short rationale in this doc.
5. **Upstream merges:** resolve core 6 files first, then rebuild distributed tools.

---

## Build

```bash
# Standard llama.cpp (no distributed tools)
cmake -B build -DLLAMA_DISTRIBUTED=OFF

# Full distributed stack
cmake -B build -DLLAMA_DISTRIBUTED=ON -DLLAMA_BUILD_TESTS=ON
cmake --build build --target orchestrator node_agent split_gen3_a split_gen3_b split_gen3_c
```
