# Task 7.5 — End-to-End Cluster Validation Suite

**Status:** ✅ **IMPLEMENTED** (pending full-cluster run on a machine with enough RAM)

## Goal

Provide a fully automatic end-to-end test that brings up the whole cluster,
installs a model, creates a session, runs distributed generation and verifies the
correctness of the system as a single chain — not as isolated components:

```
User
 → Orchestrator
 → Model Catalog
 → Install Manager
 → Node Agents
 → Split Workers
 → Distributed Inference
 → Response
```

Everything is started by the test itself. No manual launching of `orchestrator`
or `node_agent` is required.

## ✅ What was built

### New binaries

| Binary | Purpose |
| --- | --- |
| `test-cluster-e2e` | Full 10-stage end-to-end validation (main entry point) |
| `test-cluster-e2e-install-reuse` | Verifies a repeated install does not re-download |
| `test-cluster-e2e-node-loss` | Kills a node, expects `POST /session/create` → `HTTP 503` |
| `test-cluster-e2e-orchestrator-restart` | Verifies state survives an orchestrator restart |

All four live in `tools/distributed/` and share `tools/distributed/e2e_common.h`.

### Production changes required by the suite

1. **Orchestrator `GET /models` now aggregates from nodes** (was a stub returning
   `[]`). It queries every online node's `GET /models/local` and merges results by
   `model_id`:

   ```json
   [
     {
       "id": "llama-3.2-1b",
       "status": "ready",
       "ready_nodes": 3,
       "total_nodes": 3,
       "nodes": [
         { "node_id": "node-a", "ready": true, "status": "ready", "local_path": "..." }
       ]
     }
   ]
   ```

2. **Node agent heartbeat (periodic re-registration).** Each node re-registers with
   the orchestrator every 5s. This lets the cluster recover automatically after an
   **orchestrator restart** without restarting the nodes (registration is upserted
   by `node_id`).

3. **Per-node isolated model store.** New `--models-dir DIR` flag (and `MODELS_DIR`
   env) on `node_agent`, so several nodes on one host don't fight over the same
   store path during a local test.

4. **Local-source fast path for install.** If the requested catalog file is already
   present locally (e.g. the GGUF the node already runs with), the node seeds its
   store via a symlink instead of downloading from HuggingFace. This keeps the E2E
   run **offline and fast** while still exercising the full install pipeline
   (`POST /models/install` → store → `GET /models/local` ready → orchestrator
   `GET /models` ready). If no local copy exists, it falls back to a real download.

## 🏗️ Architecture

```
test-cluster-e2e (one process)
├─ spawns orchestrator                  (logs/e2e/orchestrator.log)
├─ spawns node-a / node-b / node-c      (logs/e2e/node-*.log, each --models-dir)
├─ drives the orchestrator HTTP API     (install / session / generate)
├─ runs an in-process split baseline    (split_gen3_a/b/c) as the reference
└─ writes logs/e2e/e2e-summary.json + prints an E2E REPORT
```

## 🔬 Test stages (`test-cluster-e2e`)

| # | Stage | Checks |
| --- | --- | --- |
| 1 | Environment validation | binaries present, GGUF found, ports free → `FAIL: environment validation` |
| 2 | Start cluster | orchestrator + 3 nodes, `GET /nodes` == 3 → `FAIL: node registration` |
| 3 | Model install | `POST /models/install`, poll job to `ready`, `GET /models` ready, ready on every node |
| 4 | Session create | `session_id` set, layout non-empty, pipeline = 3 nodes, layer coverage `[0, n_layer)` contiguous (no gaps/overlaps, `layer_start < layer_end`) |
| 5 | Distributed generation | `HTTP 200`, `count == max_tokens`, `tokens.size() == max_tokens`, text non-empty |
| 6 | Reference generation | in-process 3-way split baseline (same prompt, greedy sampler) |
| 7 | Token comparison | exact match of generated content (see note below) → `FAIL: token mismatch` |
| 8 | Restart recovery | restart orchestrator, nodes reconnect via heartbeat, `GET /models` restored |
| 9 | Recreate session | new session + generation succeeds |
| 10 | Graceful shutdown | all processes terminated, no leftovers |

### Note on Stage 7 (token identity)

The reference is the in-process **3-way split baseline** (the same `split_gen3_a/b/c`
workers used by the cluster, in the same `[0,5),[5,10),[10,16)` split). The
distributed pipeline and the baseline are numerically identical for all real
generated content. They can only diverge at the **end-of-generation decision**,
where the model is at a near-tie (emit `<|eot_id|>` vs continue) and a sub-ULP
difference can flip the argmax.

The suite therefore requires an **exact** match for every token **before** the
reference's first end-of-generation token (the real generated content). Tokens at
or after the model decided to stop are undefined and not compared. If the reference
never stops within `max_tokens`, all tokens are compared. The report prints e.g.
`26/26 tokens identical (up to EOS)`.

## 🛠️ How to build

> ⚠️ Building llama.cpp is memory-hungry. On very small machines (≈ small mini-PCs)
> the compile can exhaust RAM and drop your SSH session. Build on a machine with a
> few GB of free RAM, or lower parallelism (see Troubleshooting).

```bash
cd node-agent
./build.sh e2e
```

This produces (in `llama.cpp/build/bin/`):

```
orchestrator  node_agent  split_gen3_a  split_gen3_b  split_gen3_c
test-cluster-e2e
test-cluster-e2e-install-reuse
test-cluster-e2e-node-loss
test-cluster-e2e-orchestrator-restart
```

## ▶️ How to run

### Prerequisites

- A Llama 3.2 1B Q4_K_M GGUF available locally. The test auto-discovers it via:
  1. `--gguf /path/to/model.gguf` (CLI override), or
  2. `MODEL=/path/to/model.gguf` (env), or
  3. `~/models/llama-3.2-1b-instruct-q4_k_m.gguf` (default).
- Run from the `llama.cpp` directory so that `state/` and `logs/e2e/` are created
  next to `build/` (the orchestrator stores its catalog in `state/catalog.json`).

### Main suite

```bash
cd node-agent/llama.cpp

# Simplest form (auto-discovers the GGUF):
./build/bin/test-cluster-e2e

# With explicit options:
./build/bin/test-cluster-e2e --model llama-3.2-1b --prompt "Tell me a joke"

# Point at a specific GGUF and change generation length:
MODEL=~/models/llama-3.2-1b-instruct-q4_k_m.gguf \
  ./build/bin/test-cluster-e2e --max-tokens 32
```

CLI flags:

| Flag | Default | Meaning |
| --- | --- | --- |
| `--model ID` | `llama-3.2-1b` | catalog model id to install / serve |
| `--prompt STR` | `Tell me a joke` | prompt for distributed + reference generation |
| `--max-tokens N` | `32` | number of tokens to generate |
| `--gguf PATH` | (auto) | explicit GGUF file path override |

Exit code is `0` on PASS, `1` on FAIL.

### Scenario binaries

```bash
cd node-agent/llama.cpp

# Repeated install must not re-download:
./build/bin/test-cluster-e2e-install-reuse

# Losing a node must make session create fail with HTTP 503:
./build/bin/test-cluster-e2e-node-loss

# Orchestrator state survives a restart:
./build/bin/test-cluster-e2e-orchestrator-restart
```

Each prints a single `PASS`/`FAIL` line and returns the matching exit code.

## 📤 Output

- `logs/e2e/orchestrator.log`, `logs/e2e/node-a.log`, `node-b.log`, `node-c.log`
  — full child-process output.
- `logs/e2e/e2e-summary.json` — machine-readable per-stage results.
- Console `E2E REPORT`:

```
==================== E2E REPORT ====================
  Environment                  PASS  -- /home/you/models/llama-3.2-1b-instruct-q4_k_m.gguf
  Node registration            PASS  -- 3/3 nodes
  Model install                PASS  -- model ready on 3 nodes
  Session create               PASS  -- layers [0,16)
  Distributed generation       PASS  -- 32 tokens
  Token comparison             PASS  -- 26/26 tokens identical (up to EOS)
  Restart recovery             PASS  -- nodes + model restored
  Recreate session             PASS  -- regeneration ok
  Cleanup                      PASS  -- no leftover processes
---------------------------------------------------
  OVERALL RESULT: PASS
===================================================
```

The scenario binaries write their own sub-folders: `logs/e2e/install-reuse/`,
`logs/e2e/node-loss/`, `logs/e2e/orch-restart/`.

## 🧯 Troubleshooting

- **SSH drops / OOM during build.** llama.cpp's compile is RAM-heavy. Build with
  fewer jobs, e.g. edit `JOBS` in `build.sh` or run
  `cmake --build llama.cpp/build --target test-cluster-e2e -j2`. Prefer a machine
  with ≥ 4–8 GB free RAM.
- **`FAIL: environment validation` (port busy).** A previous cluster is still
  running, or another process holds the test ports. The main suite uses a base
  port `26000 + (pid % 400)` and the next 3 ports; scenarios use `27000/28000/29000`
  ranges. Kill stale `orchestrator`/`node_agent` processes and retry.
- **`FAIL: node registration`.** Check `logs/e2e/node-*.log`. The first benchmark
  run can take a while; the suite waits up to ~80s. Re-running uses a cached
  benchmark score and is fast.
- **`FAIL: token mismatch`.** Inspect the `TOKEN MISMATCH index=... expected=... actual=...`
  line on stderr; a mismatch *before* the reference's EOS indicates a real
  regression in the distributed pipeline.
- **GGUF not found.** Set `MODEL=/abs/path/to/llama-3.2-1b-instruct-q4_k_m.gguf`
  or pass `--gguf`.

## 📁 Files added / modified

**New files:**
- `tools/distributed/e2e_common.h` — shared E2E helpers (spawn+logging, HTTP,
  env validation, install/poll, layout validation, split baseline, summary).
- `tools/distributed/test-cluster-e2e.cpp` — main 10-stage suite.
- `tools/distributed/test-cluster-e2e-install-reuse.cpp`
- `tools/distributed/test-cluster-e2e-node-loss.cpp`
- `tools/distributed/test-cluster-e2e-orchestrator-restart.cpp`
- `docs/task7_5_e2e_validation.md` — this document.

**Modified files:**
- `tools/distributed/orchestrator.cpp` — aggregating `GET /models`.
- `tools/distributed/node_agent.cpp` — heartbeat re-registration, `--models-dir`/
  `MODELS_DIR`, local-source fast path for install.
- `tools/distributed/model_catalog.h` / `model_catalog.cpp` — `set_models_dir()`.
- `tools/distributed/CMakeLists.txt` — four E2E targets (built outside
  `LLAMA_BUILD_TESTS`).
- `build.sh` — new `e2e` build mode.

## 🎯 Acceptance criterion

A single command

```bash
./build/bin/test-cluster-e2e
```

validates, without manual intervention: orchestrator, node registration, model
install, model persistence, session creation, distributed inference, token
identity, restart recovery and cleanup — and writes `logs/e2e/*` plus
`e2e-summary.json`. The process exit code reflects PASS/FAIL.
