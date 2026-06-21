# Vision: Distributed Split Inference for llama.cpp

## Problem

Running large language models requires significant compute and memory. A single machine may not have enough GPU RAM for a full model, or may underutilize heterogeneous hardware (fast GPU + slow CPU, multiple nodes with different capabilities).

## Goal

Enable **layer-wise split inference** in llama.cpp: different processes (eventually different machines) each own a subset of transformer layers, exchange hidden states over the network, and together produce **bitwise-identical output** to a single-process full-model run.

## Principles

1. **Correctness first** - split paths must match full-model token IDs and text before optimizing for speed.
2. **Incremental proof** - validate each capability in isolation: partial forward, injection, TCP transport, autoregressive generation, multi-stage pipelines.
3. **No premature orchestration** - early tasks use localhost TCP, fixed layouts, and manual process forking. Scheduling and discovery come later.
4. **Independent KV caches** - each node maintains its own KV cache locally. KV is not synchronized over the network in the current design.
5. **Reuse llama.cpp infrastructure** - extend existing context/graph APIs rather than building a parallel runtime.

## Test Model

All validation uses:

- **Model:** Llama-3.2-1B-Instruct-Q4_K_M (GGUF)
- **n_layer:** 16
- **n_embd:** 2048
- **Default split (2-stage):** layer 8

Example path:

```
~/.cache/huggingface/hub/models--hugging-quants--Llama-3.2-1B-Instruct-Q4_K_M-GGUF/snapshots/.../llama-3.2-1b-instruct-q4_k_m.gguf
```

## Current Status (Tasks 1-4)

| Capability | Status |
|------------|--------|
| Partial forward pass (layer range) | Done |
| Hidden state extraction | Done |
| Hidden state injection | Done |
| Localhost TCP transport | Done |
| 2-stage autoregressive generation | Done |
| 3-stage autoregressive generation | Done |
| Per-layer cost profiling | Done |
| Orchestrator / scheduler | Not started |
| WAN / TLS / KV sync over network | Out of scope |

## Success Criteria (Global)

For any split configuration under test:

- Generated token IDs match full-model reference exactly.
- Generated text is byte-for-byte identical.
- Deterministic sampling: `temperature=0`, `top_k=1`, `top_p=1.0`, fixed seed.

## Non-Goals (Current Phase)

- Multi-node deployment and service discovery
- NAT traversal, TLS, WAN networking
- KV cache synchronization across nodes
- Production orchestration or load balancing
- Upstream llama.cpp PR submission (private fork)
