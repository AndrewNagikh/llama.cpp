# Architecture: Split Inference Pipeline

## Overview

```
Controller
    |
    v
 Node A  (layers [0, a_end))     -- own KV cache
    | hidden state [n_tokens x n_embd]
    v  TCP
 Node B  (layers [a_end, b_end)) -- own KV cache  [optional middle stage]
    | hidden state
    v  TCP
 Node C  (layers [b_end, n_layer)) -- own KV cache
    | logits
    v
 sample next token -> return to controller -> repeat
```

Two-stage variant (Task 3): A -> B, where B is the final node and samples.

## Core API Extensions

Staging header: `src/llama-ext.h`

| API | Purpose |
|-----|---------|
| `llama_set_layer_range(ctx, start, end)` | Run only layers `[start, end)`. `end < 0` means `n_layer`. |
| `llama_set_hidden_state(ctx, data, n_tokens)` | Inject hidden state when `layer_start > 0`. Layout: `[n_embd * n_tokens]`, row-major per token. |
| `llama_clear_hidden_state(ctx)` | Clear injected hidden state. |
| `llama_get_embeddings(ctx)` | Read hidden state output when `layer_end < n_layer`. |

### Layer range behavior

- `[0, layer_end)` with `layer_end < n_layer`: skip output norm + LM head; output **hidden state** via `llama_get_embeddings()`.
- `[layer_start, n_layer)`: normal **logits** output (includes output norm + LM head when `layer_start > 0` requires injection first).

### Modified source files

| File | Change |
|------|--------|
| `src/llama-cparams.h` | `layer_start`, `layer_end` in context params |
| `src/llama-context.cpp/h` | API, decode logic, hidden state buffer |
| `src/models/llama.cpp` | Layer loop `[layer_start, layer_end)` |
| `src/llama-graph.h/cpp` | Graph reuse checks layer range; `build_inp_hidden()` |
| `common/arg.cpp`, `common/common.cpp/h` | CLI `--layer-start`, `--layer-end` |

## TCP Wire Protocol

Files: `tests/split_tcp_wire.h`, `tests/split_tcp_wire.cpp`

### Hidden state message (prefill / transfer)

```
split_tcp_header  (20 bytes): magic 'SPTH', version, n_tokens, n_embd, layer_end
split_gen_hidden_meta: pos_start, include_logits
payload: float[n_tokens * n_embd]
```

### Controller <-> Node A (generation)

```
split_gen_a_req:  cmd, n_tokens, pos_start, layer_end, include_logits, [token ids...]
split_gen_a_resp: token_id, timing fields, [optional logits]
```

Commands: `RESET`, `PREFILL`, `DECODE`, `SHUTDOWN`.

### Node A <-> Node B (2-stage)

```
split_ab_cmd: RESET | HIDDEN | SHUTDOWN
split_gen_b_resp: token_id, ms_compute, [logits]
```

### 3-node extensions (Task 4)

```
split_gen3_c_resp:   token_id, ms_compute, ms_sample
split_gen3_mid_resp: token_id, ms_b_compute, ms_bc_xfer, ms_c_compute, ms_c_sample
split_gen3_a_resp:   full timing chain back to controller
```

## Process Layout

### 2-stage (Task 3)

| Binary | Role |
|--------|------|
| `split_gen_a` | Layers `[0, layer_end)`, forwards hidden to B |
| `split_gen_b` | Layers `[layer_start, n_layer)`, samples token |
| `test-autoregressive-split` | Controller: fork, drive loop, compare vs full model |

Startup order: B listens -> A connects to B, listens for controller -> controller connects to A.

### 3-stage (Task 4)

| Binary | Role |
|--------|------|
| `split_gen3_a` | Entry node |
| `split_gen3_b` | Middle node (forwards hidden to C) |
| `split_gen3_c` | Final node (samples token) |
| `test-autoregressive-split-3node` | Controller for layouts A/B/C |

Startup order: C listens -> B connects to C, listens -> A connects to B, listens -> controller connects to A.

## KV Cache and Positions

Each process keeps an **independent** KV cache for its layer range only.

Position scheme (autoregressive):

- Prefill: positions `0 .. n_prompt-1`
- Generation step `k`: position `n_prompt + k - 1` for the token being decoded

## Critical Implementation Patterns

### Batch initialization

Always use `llama_batch_init()` when setting `batch.logits[i]`. Never write to null from `llama_get_one()`.

### Prefill hidden extraction (entry node)

Mark **all prompt tokens** as output (`batch.logits[i]=1`) to capture full hidden state for downstream injection.

### Middle node prefill (3-stage)

Process tokens **one at a time** through middle layers. Multi-output on partial layer ranges triggers GGML assert; single-token decode avoids this while correctly populating KV.

### Final node prefill

Use single-batch decode with `last_only=true` (all tokens in batch, only last marked as output) - same pattern as `test-injection-proof`.

### Fork safety (macOS / Metal)

Controller must **not** hold an active Metal/GPU context across `fork()`.

Pattern:

1. Tokenize prompt (load model briefly, free before fork), OR run split paths first.
2. Fork child processes (each loads Metal independently).
3. Run split generation.
4. Load model fresh for full-model reference comparison.
5. Compare tokens **before** freeing model/vocab.

## Shared Test Helpers

`tests/split_gen_common.h`:

- Sampler chain: `top_k=1`, `top_p=1.0`, `temp=0.0`, greedy, seed 1234
- `split_gen_decode_tokens()`, `split_gen_decode_one()`, `split_gen_decode_hidden()`
- Prompt: `"Tell me a joke"`

## Build Targets

```bash
cmake -B build -DLLAMA_BUILD_TESTS=ON
cmake --build build --target \
  test-partial-forward test-injection-proof test-split-tcp \
  split_gen_a split_gen_b test-autoregressive-split \
  split_gen3_a split_gen3_b split_gen3_c test-autoregressive-split-3node \
  benchmark-layer-cost -j8
```

Binaries land in `build/bin/`.

## Known Limitations

- `llama-cli` partial mode may crash (sampler expects logits when partial graph has none).
- Layer cost benchmark memory column reflects full model allocation, not incremental per-layer delta.
- TCP overhead is negligible on localhost; multi-process split is slower than single-process full model.
- No recovery from node failure; no reconnection protocol.
