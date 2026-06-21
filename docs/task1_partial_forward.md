# Task 1: Partial Forward Pass

## Goal

Run only a subset of transformer layers `[layer_start, layer_end)` for Llama models, producing either hidden state or logits depending on the range.

## API

```c
// src/llama-ext.h
void llama_set_layer_range(struct llama_context * ctx, int32_t start, int32_t end);
// end < 0 means n_layer (full model)
```

CLI flags: `--layer-start`, `--layer-end` (via `common/arg.cpp`).

## Behavior

| Range | Output |
|-------|--------|
| `[0, n_layer)` | Normal logits (full model) |
| `[0, k)` where `k < n_layer` | Hidden state via `llama_get_embeddings()`, no output norm / LM head |
| `[k, n_layer)` | Requires hidden injection (Task 2); produces logits |

## Implementation

- `src/llama-cparams.h`: stores `layer_start`, `layer_end`
- `src/models/llama.cpp`: layer loop iterates `[layer_start, layer_end)` only
- `src/llama-graph.h`: graph reuse invalidates when layer range changes
- `src/llama-context.cpp`: skips LM head when `layer_end < n_layer`

## Test

**File:** `tests/test-partial-forward.cpp`

Validates multiple layer ranges against full-model output for Llama-3.2-1B.

```bash
./build/bin/test-partial-forward /path/to/model.gguf
```

Expected: `test-partial-forward: OK (n_layer=16 n_embd=2048)`

## Bug Fixes During Development

1. **`common/common.cpp`**: added `#include "../src/llama-ext.h"` for CLI partial mode.
2. **Batch bug in test**: `llama_batch_get_one()` leaves `logits=nullptr`. Fixed with `llama_batch_init()` and explicit `batch.logits[i]` setup.

## Status

**DONE** - partial forward validated for prefill and single-token decode.

## Known Issue

CLI partial mode (`llama-cli --layer-end 8`) may still crash because the sampler expects logits when the partial graph outputs embeddings only. Test binaries handle this correctly; CLI integration is incomplete.
