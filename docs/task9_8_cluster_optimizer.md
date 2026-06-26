# Task 9.8 — Cluster Optimizer & Automatic Rebalancing

## Goal

The Cluster Optimizer decides whether the cluster should change a model's **Desired Layout** when nodes join, leave, or change performance/memory. It does **not** download data or run synchronization itself.

```
Node change → Cluster Optimizer → Performance simulation → Decision
    → (if REBALANCE) Pending Layout → Install Plan → Sync Engine → Commit Layout
```

## Modules

| Module | Path | Role |
|--------|------|------|
| `layout_estimator` | `orchestrator/layout_estimator/` | Pipeline throughput + rebalance byte cost |
| `cluster_optimizer` | `orchestrator/optimizer/` | Node roles, policy, REBALANCE / KEEP / STORAGE_ONLY |

## Node roles

- **ACTIVE** — participates in inference (prefill + decode).
- **STORAGE_ONLY** — stores layers, may help node-to-node transfer; excluded from layout planner ACTIVE set.
- **STANDBY** — offline or unused.

Weak nodes with enough RAM but score below `median × storage_only_score_ratio` (default 0.25) become STORAGE_ONLY.

## Decision policy (`optimizer_policy`)

| Parameter | Default |
|-----------|---------|
| `min_decode_improvement_percent` | 5% |
| `min_prefill_improvement_percent` | 5% |
| `max_rebalance_cost_bytes` | unlimited |
| `allow_storage_only_nodes` | true |

Rebalance is recommended when decode **or** prefill gain meets the threshold and cost is within the cap.

## Two-phase layout switch

1. Optimizer stores **candidate** layout in `pending_layout` (current layout unchanged).
2. Install plan is built against `pending_layout`.
3. Synchronization Engine executes transfers.
4. Coverage is checked against **pending** layout.
5. On `READY`, `commit_pending_layout()` atomically promotes pending → current.
6. On failure, `discard_pending_layout()`.

Active inference sessions keep using the committed layout until step 5 completes.

## Registry fields

- `optimization` — last optimizer run (`optimization_result`).
- `pending_layout` — candidate layout during rebalance.

## REST API

### `POST /models/{id}/optimize`

Runs optimizer. Optional JSON body overrides policy fields.

Response example:

```json
{
  "decision": "REBALANCE",
  "current_decode_tps": 420,
  "candidate_decode_tps": 475,
  "rebalance_cost_bytes": 1342177280,
  "better": true
}
```

If decision is `REBALANCE`, rebalance pipeline starts asynchronously.

### `GET /models/{id}/optimization`

Returns the stored last result.

## Automatic triggers

Optimizer runs in a background thread when:

- a node registers (new node, benchmark change ≥5%, memory change ≥10%);
- manifest is built;
- layout is built.

## Tests

**Unit:** `test-layout-estimator`, `test-storage-only-node`, `test-performance-simulation`, `test-optimizer-policy`, `test-no-benefit`, `test-improvement-threshold`

**Integration:** `test-node-join`, `test-slow-node`, `test-node-loss`, `test-memory-change`

**E2E:** `test-cluster-e2e-optimizer-api`, `test-cluster-e2e-optimizer-join`

## Out of scope (Task 9.8)

- Inference runtime changes
- Distributed prefill / KV cache
- Optimizer does not replace Task 8 session planner (`POST /session/create`)
