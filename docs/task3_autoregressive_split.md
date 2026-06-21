# Task 3: Autoregressive Split Generation (2-Stage)

## Goal

Prove split inference works for **autoregressive generation**, not just prefill. Generate multiple tokens with independent KV caches and verify token-for-token equivalence with full-model generation.

## Architecture

```
Controller
    |
    v
Process A  layers [0, 8)     own KV
    | hidden state (TCP)
    v
Process B  layers [8, 16)    own KV
    | sample token
    v
Controller (repeat for each token)
```

## Files

| File | Role |
|------|------|
| `tests/split_gen_a.cpp` | Entry daemon: decode, forward hidden to B |
| `tests/split_gen_b.cpp` | Final daemon: inject, decode, sample |
| `tests/test-autoregressive-split.cpp` | Controller |
| `tests/split_gen_common.h` | Sampler, decode helpers, constants |
| `tests/split_tcp_wire.h/cpp` | Gen protocol extensions |

## Generation Protocol

Controller <-> A:

| Command | Action |
|---------|--------|
| `RESET` | Clear KV on A and B |
| `PREFILL` | Send prompt tokens, return first sampled token |
| `DECODE` | Send 1 token at position, return next sampled token |
| `SHUTDOWN` | Tear down |

A <-> B: reuse `split_ab_cmd` + hidden payload + `split_gen_b_resp`.

## Test Parameters

| Parameter | Value |
|-----------|-------|
| Prompt | `"Tell me a joke"` |
| temperature | 0.0 |
| top_k | 1 |
| top_p | 1.0 |
| seed | 1234 |
| max_new_tokens | 16 |
| split_layer | 8 |

## Validation

Reference: standard full-model generation in same process (after split path completes).

Success: all 16 token IDs match; generated text byte-for-byte identical.

Example output (both paths):

```
 about a cat.
Why did the cat join a band?
Because it wanted to
```

## Diagnostics

Per step:

```
step=N token=ID text=...
```

On mismatch: step index, full vs split token, optional `max_abs_diff(logits)`.

## Performance (example, M1 Pro)

```
perf: full_ms=169.790 split_ms=286.432
perf: full_tps=94.234 split_tps=55.860
perf: split_a_ms=9.850 split_b_ms=176.976 split_tcp_ms=0.301
```

TCP transfer is negligible; overhead is from dual-process inference and per-token IPC.

## Fork Safety Fix

Loading Metal in the parent before `fork()` caused child segfaults (exit 139) on macOS.

Fix:

1. Parent tokenizes prompt only (brief model load, then free).
2. Run split path first (children load GPU independently).
3. Load model fresh for full reference after `waitpid`.
4. Compare before freeing vocab.

## Run

```bash
./build/bin/test-autoregressive-split /path/to/model.gguf
```

Expected: `test-autoregressive-split: OK (16 tokens)`, `MATCH = TRUE`

## Status

**DONE**
