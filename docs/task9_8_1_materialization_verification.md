# Task 9.8.1 — GGUF Materialization Verification & Inference Parity

Prove that a GGUF materialized from the Layer Store is functionally equivalent to the original GGUF file.

## Verification Pipeline

The pipeline runs in `verification/verification_pipeline.cpp` and is exposed via:

- CLI tools (see below)
- `POST /models/{id}/verify` on the orchestrator
- `GET /models/{id}/verify` for the last report

Stages (in order):

1. **Materialize** — assemble a full GGUF from Layer Store + Manifest
2. **Header / Metadata / Tensor directory** — structural diff (`gguf_diff`)
3. **Metadata KV prefixes** — `tokenizer.*`, `general.*`, `rope.*`, `chat_template.*`, `special_vocab.*`, `quantization.*`
4. **Tensor checksums** — per-tensor SHA256 (`tensor_verify`)
5. **Alignment** — tensor/data alignment and padding
6. **Layer store** — each blob matches manifest byte ranges
7. **Repeatability** — two consecutive materializations produce identical SHA256
8. **Logits parity** — last-token logits vs original (numerical tolerance)
9. **Sampling parity** — greedy decoding: first token, 8 tokens, 32 tokens

Example report:

```json
{
  "model_id": "llama-3.2-1b",
  "passed": true,
  "header": "OK",
  "metadata": "OK",
  "tensor_directory": "OK",
  "tensor_checksums": "OK",
  "alignment": "OK",
  "layer_store": "OK",
  "materialization_repeatability": "OK",
  "logits": "OK",
  "sampling": "OK"
}
```

## Module layout

```
tools/distributed/verification/
  gguf_inspector.*      # JSON GGUF structure dump
  gguf_diff.*           # Header/metadata/tensor-directory diff
  tensor_verify.*       # Per-tensor SHA256
  layer_verify.*        # Layer Store vs manifest ranges
  metadata_verify.*     # KV prefix comparison
  alignment_verify.*    # Alignment rules
  parity_verify.*       # Logits + greedy sampling
  materialization_verify.*  # Repeatability + full materialize helper
  verification_pipeline.*   # Orchestrates all checks
  worker_verify.*       # Worker startup checks (--verify-materialization)
  layer_store_populate.*    # Test helper: populate store from local GGUF
```

## CLI tools

| Tool | Purpose |
|------|---------|
| `gguf_inspect PATH` | Print version, alignment, metadata size, tensor count, offsets, sizes, names, types (JSON) |
| `gguf_diff ORIG MAT` | Compare header, metadata, tensor directory; print all differences |
| `tensor_verify ORIG MAT` | Per-tensor name, offset, size, dtype, SHA256 |
| `layer_verify STORE MANIFEST SOURCE` | Verify each Layer Store blob against manifest ranges |

Build targets are registered in `tools/distributed/CMakeLists.txt`.

## Worker runtime verification

`node_agent` accepts `--verify-materialization`. Before assembling a worker GGUF, it checks:

- metadata size matches manifest
- embedding / layer / output blob presence and checksums for the worker's layer range

On failure, worker startup is aborted.

## Inference parity

Prompt (default): `The capital of France is`

1. Run original and materialized models with identical greedy sampling
2. Compare last-token logits (float tolerance via `verification_common`)
3. Compare greedy token sequences (must match exactly)

## Byte-for-byte reproducibility

`verify_materialization_repeatability()` materializes twice into separate files and compares SHA256. Any mismatch indicates non-deterministic serialization, alignment, or write order.

## Multi-model regression

`test-verification-multi-model` runs the full pipeline on several GGUF files:

- `MODEL` — primary model (required for CI skip code 77 if missing)
- `VERIFY_MODELS` — comma-separated list of additional paths
- Default search: `~/models/llama-3.2-…`, `qwen2.5-…`, `gemma-2-…` (skip if absent)

Use this when upgrading llama.cpp to catch GGUF format or loader regressions early.

## Tests

### Unit / integration (require `MODEL` env or default path)

| Test | Checks |
|------|--------|
| `test-gguf-inspector` | Inspector JSON fields |
| `test-gguf-diff` | Original vs materialized structure |
| `test-tensor-checksum` | Tensor SHA256 |
| `test-layer-checksum` | Layer blob checksums |
| `test-alignment` | Alignment on materialized file |
| `test-materialization-repeatability` | Double materialize SHA256 |
| `test-materialized-load` | llama.cpp loads materialized GGUF |
| `test-logit-parity` | Last-token logits |
| `test-sampling-parity` | Greedy tokens (32) |
| `test-layer-store-verification` | Layer store + full pipeline |
| `test-verification-multi-model` | Regression across multiple GGUF |

### E2E

`test-cluster-e2e-verification` — single-node cluster:

```
Register → Manifest → Layout → Sync → Verification PASS
  → delete blob → Verification FAIL → repair → Verification PASS
```

Orchestrator and node share `--models-dir` so `POST /verify` sees the synced Layer Store.

## Constraints (Task 9.8.1)

Do not change Planner, Synchronization Engine, KV Cache, or Distributed Prefill except to fix bugs found by this verification framework.

## Running locally

```bash
export MODEL=/path/to/model.gguf
cmake --build build --target test-gguf-inspector test-gguf-diff test-logit-parity
./build/bin/test-gguf-inspector
./build/bin/test-logit-parity
```

Optional multi-model:

```bash
export VERIFY_MODELS=/path/llama.gguf,/path/qwen.gguf
./build/bin/test-verification-multi-model
```
