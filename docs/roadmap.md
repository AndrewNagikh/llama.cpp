# Roadmap: Split Inference

## Completed

### Task 1 - Partial Forward
- [x] `llama_set_layer_range()` API
- [x] Hidden state output when `layer_end < n_layer`
- [x] `test-partial-forward`

### Task 2 - Injection and TCP
- [x] `llama_set_hidden_state()` / `llama_clear_hidden_state()`
- [x] `test-injection-proof` (max_abs_diff = 0.0)
- [x] TCP wire protocol (`split_tcp_wire`)
- [x] `split_sender` / `split_receiver` / `test-split-tcp`

### Task 3 - 2-Stage Autoregressive Generation
- [x] `split_gen_a` / `split_gen_b`
- [x] Gen protocol (RESET, PREFILL, DECODE, SHUTDOWN)
- [x] `test-autoregressive-split` (16 tokens, MATCH = TRUE)
- [x] Per-step diagnostics and basic perf metrics

### Task 4 - 3-Stage Pipeline and Profiling
- [x] `split_gen3_a/b/c`
- [x] `test-autoregressive-split-3node` (3 layouts, 32 tokens each)
- [x] Per-node timing (A, AB, B, BC, C, sample)
- [x] `benchmark-layer-cost` + `tests/layer_cost.csv`
- [x] Layout TPS comparison report

---

## Next Steps (Suggested)

### Task 5 - Orchestrator Prototype
- [ ] Read `layer_cost.csv` and node capability descriptors
- [ ] Given N heterogeneous nodes, compute optimal layer partition
- [ ] Dynamic process spawn (still localhost first)
- [ ] Reconfigure pipeline without manual layout constants

### Task 6 - Multi-Machine TCP
- [ ] Replace `127.0.0.1` with configurable host addresses
- [ ] Connection retry and basic health checks
- [ ] Larger hidden-state payloads (batching, compression - optional)

### Task 7 - Production Hardening
- [ ] Graceful shutdown and error propagation through pipeline
- [ ] Reconnection protocol after node failure
- [ ] TLS for WAN (if needed)
- [ ] Fix `llama-cli` partial mode sampler crash

### Task 8 - Larger Models
- [ ] Validate on 7B+ models where split inference has real memory benefit
- [ ] Measure memory per layer range (not just full model allocation)
- [ ] Quantization interaction with split paths

---

## Out of Scope (Explicit)

- KV cache synchronization over network
- NAT traversal
- Service discovery / distributed deployment
- Upstream llama.cpp PR (private fork; follow AGENTS.md if contributing upstream)

---

## Test Matrix (Regression)

Run after any change to split inference code:

```bash
MODEL=/path/to/llama-3.2-1b-instruct-q4_k_m.gguf

./build/bin/test-partial-forward "$MODEL"
./build/bin/test-injection-proof "$MODEL"
./build/bin/test-split-tcp "$MODEL"
./build/bin/test-autoregressive-split "$MODEL"
./build/bin/test-autoregressive-split-3node "$MODEL"
./build/bin/benchmark-layer-cost "$MODEL" --csv tests/layer_cost.csv
```

All tests should pass on Unix (fork-based controllers require macOS/Linux).

---

## Key Files Reference

```
src/llama-ext.h              API staging header
src/llama-cparams.h          layer_start, layer_end
src/models/llama.cpp         partial layer loop
tests/split_tcp_wire.*       TCP protocol
tests/split_gen_common.h     shared sampler/decode helpers
tests/split_gen{3,}_*.cpp    process daemons
tests/test-autoregressive*   controllers
tests/benchmark-layer-cost.cpp
tests/layer_cost.csv         profiling output
docs/                        this documentation set
```
