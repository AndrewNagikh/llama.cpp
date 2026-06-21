# Task 2: Hidden State Injection and TCP Transport

## Task 2A: Hidden State Injection

### Goal

When `layer_start > 0`, bypass token embedding lookup and inject hidden state from a prior pipeline stage.

### API

```c
void llama_set_hidden_state(struct llama_context * ctx, const float * data, int32_t n_tokens);
void llama_clear_hidden_state(struct llama_context * ctx);
```

Layout: `[n_embd * n_tokens]`, row-major per token (same as `batch.embd`).

### Implementation

- `llm_graph_input_hidden` + `build_inp_hidden()` in `src/llama-graph.h/cpp`
- `src/models/llama.cpp`: if `layer_start > 0` use `build_inp_hidden()`, else `build_inp_embd()`
- Context stores `hidden_state_inp` buffer; copied on `llama_set_hidden_state()`

### Test

**File:** `tests/test-injection-proof.cpp`

Pipeline: full forward vs split (layers 0-7 -> inject -> layers 8-15).

```bash
./build/bin/test-injection-proof /path/to/model.gguf
```

Result: `max_abs_diff = 0.0` (bitwise-equal logits).

### Status

**DONE**

---

## Task 2B: Multi-Process TCP Prefill

### Goal

Two processes on localhost exchange hidden state over TCP and produce logits matching the full model.

### Files

| File | Role |
|------|------|
| `tests/split_tcp_wire.h/cpp` | Wire protocol + TCP helpers |
| `tests/split_sender.cpp` | Process A: layers 0-7, sends hidden |
| `tests/split_receiver.cpp` | Process B: receives hidden, injects, layers 8-15, logits |
| `tests/test-split-tcp.cpp` | Forks both, compares with full model |

### Wire format

```
Header: split_tcp_header (magic SPTH, n_tokens, n_embd, layer_end)
Payload: float[n_tokens * n_embd]
```

### Test

```bash
./build/bin/test-split-tcp /path/to/model.gguf
```

Result: `test-split-tcp: OK`, `max_abs_diff = 0.0`

### Process startup

1. Receiver listens first (`listen()` before model load, then `accept()`).
2. Sender connects after model load.
3. Controller retry loop for connection (60 x 100ms).

### Status

**DONE**
