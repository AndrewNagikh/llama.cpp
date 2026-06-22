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

### Task 5 - Orchestrator + Node Agent
- [x] `orchestrator` HTTP coordinator
- [x] `node_agent` supervisor (registers, launches split_gen3 workers)
- [x] Node registration `/register`
- [x] Static pipeline layout A `[0,5) [5,10) [10,16)`
- [x] End-to-end generation `/session/generate`
- [x] `test-orchestrator-3node` (32/32 vs split baseline)
- [x] Graceful failure when middle node stops
- [x] Remote host TCP (`--b-host`, `--bc-host`, `--bind`)
- [x] Separate deploy repo `distributed-llm/node-agent/`

---

## Next Steps (Suggested)

### Task 6 - Dynamic Layer Planner
- [x] `GET /capabilities` on node agent
- [x] `--score` CLI flag
- [x] Score/hardware in registration
- [x] `layer_planner.h/cpp` proportional algorithm
- [x] Dynamic layout in `POST /session/create`
- [x] N-node pipeline configuration (not hardcoded 3)
- [x] `test-layer-planner`
- [x] `test-orchestrator-dynamic-layout`

---

## Next Steps (Suggested)

### Task 7 - Auto Benchmark
- [ ] Advertise host / NAT-friendly registration
- [ ] Connection retry and health checks between sessions
- [ ] Larger hidden-state payloads (optional compression)

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
./build/bin/test-orchestrator-3node "$MODEL"
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
