# Task 4: Three-Stage Pipeline and Layer Cost Profiling

## Goal

1. Extend split generation from 2 stages to 3 stages.
2. Collect per-node performance data for future scheduler/orchestrator design.
3. Benchmark individual layer compute cost.

## 3-Stage Architecture

```
Controller -> Node A -> TCP -> Node B -> TCP -> Node C -> sample -> repeat
```

Each node owns its layer range and maintains an independent KV cache.

## Layouts Tested

| Layout | Node A | Node B | Node C |
|--------|--------|--------|--------|
| A | [0, 5) | [5, 10) | [10, 16) |
| B | [0, 3) | [3, 8) | [8, 16) |
| C | [0, 8) | [8, 12) | [12, 16) |

Defined in `tests/split_gen3_common.h`.

## Files

| File | Role |
|------|------|
| `tests/split_gen3_a.cpp` | Entry node |
| `tests/split_gen3_b.cpp` | Middle node (forwards hidden to C) |
| `tests/split_gen3_c.cpp` | Final node (samples token) |
| `tests/test-autoregressive-split-3node.cpp` | Controller for all layouts |
| `tests/benchmark-layer-cost.cpp` | Per-layer prefill/decode benchmark |
| `tests/layer_cost.csv` | Generated profiling dataset |
| `tests/split_gen3_common.h` | Layouts, perf structs |

## Test Parameters

| Parameter | Value |
|-----------|-------|
| Prompt | `"Tell me a joke"` |
| temperature | 0.0 |
| top_k | 1 |
| top_p | 1.0 |
| seed | 1234 |
| max_new_tokens | 32 |

## Validation Results (M1 Pro, example run)

All three layouts: **32/32 tokens match**, `MATCH = TRUE`.

```
Layout      TPS
A           67.571
B           74.622   <- best
C           72.507
FULL        ~108
```

Layout B (`[0,3) [3,8) [8,16)`) had highest split throughput in this run - middle stage is relatively light (5 layers) and final stage carries the LM head cost.

## Per-Step Profiling

For each decode step the controller logs:

```
perf step=N A=... AB=... B=... BC=... C=... sample=... total=... ms
```

| Field | Meaning |
|-------|---------|
| A | Node A compute time |
| AB | TCP A->B latency (send + recv) |
| B | Node B compute time |
| BC | TCP B->C latency |
| C | Node C compute time |
| sample | Sampling time (in C) |
| total | Sum of above |

Layout summary also prints average ms/token per field.

## Middle Node Prefill Fix

**Problem:** Middle node B must export hidden states for **all** prefill tokens to C. Using `last_only=false` on partial layers triggers GGML assert (`tensor read out of bounds`). Using `last_only=true` only captures the last token's hidden state.

**Solution:** Process prefill tokens **one at a time** through B's layers, collecting each output hidden state into a buffer, then forward the full buffer to C in one message.

Final node C uses single-batch decode with `last_only=true` for multi-token prefill (same as injection proof step 2).

## Layer Cost Benchmark

```bash
./build/bin/benchmark-layer-cost /path/to/model.gguf --csv tests/layer_cost.csv
```

Measures each layer `[i, i+1)` individually:

- **prefill_ms**: prompt `"Tell me a joke"` through layer i only (setup for i>0 not timed)
- **decode_ms**: average single-token decode (10 iterations)
- **memory_bytes**: context memory breakdown total

### Sample results (M1 Pro, decode ms/token)

| Layer | Decode (ms) | Notes |
|-------|-------------|-------|
| 0 | 4.37 | Includes embedding lookup |
| 1-14 | 0.53-0.60 | Transformer blocks |
| 15 | 0.70 | Includes output norm + LM head |

Layer 0 prefill is slower (~2.6 ms) due to embedding. Memory column ~1 GB reflects full model loaded in each context, not per-layer incremental cost.

Output: console table + `tests/layer_cost.csv`.

## Run

```bash
cmake --build build --target \
  split_gen3_a split_gen3_b split_gen3_c \
  test-autoregressive-split-3node benchmark-layer-cost -j8

./build/bin/test-autoregressive-split-3node /path/to/model.gguf
./build/bin/benchmark-layer-cost /path/to/model.gguf --csv tests/layer_cost.csv
```

Expected: `test-autoregressive-split-3node: OK (32 tokens x 3 layouts)`

## Bug Fixes

1. **Use-after-free**: comparison used `vocab` after `llama_model_free()` - fixed by comparing before free.
2. **Controller connect timeout**: 3 processes loading Metal need up to 30s startup; retry loop extended to 300 x 100ms.
3. **Final node multi-output**: C must not mark all tokens as output (backend sampling allows one output per sequence).

## Status

**DONE**
