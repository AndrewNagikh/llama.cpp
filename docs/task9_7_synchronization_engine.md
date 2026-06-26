# Task 9.7 — Synchronization Engine & Layer Store

## Goal

Move from “download full GGUF to each node” to “synchronize a distributed layer store”.

After Task 9.7:

- Node Agent uses **Layer Store** as the local source of truth for installed layers.
- **Synchronization Engine** executes Install Plan operations.
- **HTTP Range Download Executor** is the first synchronization backend.
- Coverage and `/installed-layers` are driven by Layer Store state.
- Inference runtime is unchanged (still uses `--model` GGUF for workers).

## Architecture

```
Manifest → Desired Layout → Coverage → Install Plan
    → Synchronization Engine → Layer Store → Coverage READY
```

### Layer Store layout

```
~/.distributed-llm/models/{model_id}/
  manifest.json
  checksums.json
  layers/
    000.bin
    001.bin
    ...
```

Blobs are opaque byte ranges from the remote GGUF (or any future source). Layer Store does not parse GGUF internals.

### Key modules

| Module | Path |
|--------|------|
| Layer Store | `node_agent/layer_store/` |
| Sync types / jobs | `node_agent/synchronization/sync_types.*` |
| Sync engine | `node_agent/synchronization/synchronization_engine.*` |
| Executor interface | `node_agent/synchronization/synchronization_executor.h` |
| HTTP Range executor | `node_agent/synchronization/executors/http_range/` |

### Executor interface

```cpp
class synchronization_executor {
public:
    virtual executor_result execute(const install_operation &, layer_store &) = 0;
};
```

`HTTPRangeDownloadExecutor` fetches `tensor_offset` + `tensor_length` from `source_url`, writes the blob, verifies checksum, updates Layer Store.

Future executors (Node-to-Node, S3, Torrent, …) plug into the same interface.

### Job model

Each install operation progresses:

```
QUEUED → RUNNING → VERIFYING → READY
                              ↘ FAILED
```

Cluster jobs are tracked on the orchestrator; per-node jobs on each Node Agent.

## REST API

### Node Agent

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/installed-layers?model={id}` | Layer Store state only |
| `POST` | `/models/{id}/install/execute` | Run local sync job |
| `GET` | `/jobs/{id}` | Per-node job progress |

### Orchestrator

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/models/{id}/install/execute` | Dispatch install plan to nodes |
| `GET` | `/jobs/{id}` | Cluster sync job status |

Existing endpoints (`/models/{id}/install-plan`, `/coverage/*`) are unchanged.

## E2E flow

After layout:

1. `POST /models/{id}/install-plan` — build plan (DOWNLOAD ops when Layer Store empty)
2. `POST /models/{id}/install/execute` — run synchronization
3. `POST /models/{id}/coverage/refresh` — expect `READY`
4. `POST /session/create` — unchanged runtime path

Repair scenario: delete a layer blob → `PARTIAL` coverage → reconcile → install plan → execute → `READY`.

## Tests

**Unit:** `test-layer-store`, `test-layer-verify`, `test-http-range-executor`, `test-install-job`, `test-job-progress`, `test-layer-delete`

**Integration:** `test-sync-engine`, `test-partial-install`, `test-corrupted-layer`, `test-coverage-after-sync`

**E2E:** updated `test-cluster-e2e`, `test-cluster-e2e-node-loss`, `test-cluster-e2e-install-reuse`, `test-cluster-e2e-orchestrator-restart`

## Limitations (Task 9.7)

- No node-to-node transfer
- No runtime / KV / prefill changes
- Only HTTP Range (and `file://` for local tests) executor
- Checksum stub `manifest:layer:{N}` still used by install planner; verification checks size + stored checksum
