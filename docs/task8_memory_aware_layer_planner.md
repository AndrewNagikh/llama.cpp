# Task 8 — Memory-Aware Layer Planner

## Goals

Make the distributed inference orchestrator aware of heterogeneous node
memory before it builds a pipeline:

- Detect node hardware automatically (RAM, VRAM, CPU, backend).
- Estimate model memory from GGUF metadata or catalog entries.
- Decide whether a model fits the cluster *before* creating a session.
- Distribute individual layers across nodes while respecting each node's
  execution budget and leaving room for future layer sharding (Task 9).
- Never treat `free_ram + free_vram` as a single pool.

## Architecture decisions

1. **Independent budgets.**  Each node exposes `free_ram_bytes` and
   `free_vram_bytes`.  The planner's primary budget per node is GPU VRAM if a
   GPU is present, otherwise CPU RAM.  A model that requires 44 GB does not
   fit on an RTX 4070 Ti with 12 GB VRAM just because the machine also has
   32 GB RAM.

2. **Layer granularity.**  `model_memory_requirements` carries one
   `model_layer_memory` entry per layer with `weight_bytes` and
   `kv_bytes_per_token`.  The planner assigns contiguous layer ranges by
   consuming layers one at a time against each node's budget.  In Task 9 the
   same planner will accept real per-layer weights without changing its
   interface.

3. **GGUF-agnostic planner.**  The planner does not parse GGUF files; it
   receives `(model memory structure, node resources)` and returns
   `(assignments)`.  The orchestrator is responsible for reading model
   metadata before asking the planner.

## New / changed data structures

`tools/distributed/dist_common.h`:

```cpp
struct dist_node_memory {
    uint64_t total_ram_bytes  = 0;
    uint64_t free_ram_bytes   = 0;
    uint64_t total_vram_bytes = 0;
    uint64_t free_vram_bytes  = 0;
    bool     has_gpu          = false;
};

struct dist_node_cpu {
    std::string cpu_name;
    int         physical_cores = 0;
    int         logical_cores  = 0;
    uint64_t    cache_l3_bytes = 0;
};

struct dist_node_system {
    std::string os;
    std::string arch;
};
```

`dist_node_info` now embeds `memory`, `cpu` and `system` fields, and
`dist_node_capabilities` also carries byte-level memory fields for backwards
compatible JSON output.

## Memory estimator

`tools/distributed/memory_estimator.{h,cpp}`:

```cpp
struct model_layer_memory {
    int32_t  layer_index;
    uint64_t weight_bytes;
    uint64_t kv_bytes_per_token;
};

struct model_memory_requirements {
    std::string model_id;
    int32_t     n_layer;
    int32_t     n_embd;
    int32_t     n_ctx;
    std::vector<model_layer_memory> layers;

    uint64_t weights_bytes;
    uint64_t kv_bytes;
    uint64_t compute_bytes;
    uint64_t scratch_bytes;

    uint64_t total_bytes() const;
};
```

### Weights

When a local GGUF path is available, `weights_bytes` is the file size of the
model.  If only a catalog entry is available, it is `size_gb * 1024^3`.

### KV cache

The KV cache size is architecture-aware:

```
kv_bytes_per_layer = 2 * n_head_kv * head_dim * n_ctx * kv_element_bytes
kv_bytes = kv_bytes_per_layer * n_layer
```

`head_dim` is read from `${arch}.attention.key_length` metadata when present,
and falls back to `n_embd / n_head`.  This matches real llama.cpp layouts and
will be reused by Task 11.

### Compute and scratch

These are conservative, architecture-agnostic placeholders:

```cpp
compute_bytes = max(256 MiB, weights_bytes / 16);
scratch_bytes = max(128 MiB, weights_bytes / 32);
```

## Cluster feasibility check

`dist_check_cluster_memory_fit()` sums each online node's **primary**
budget (VRAM if GPU, else RAM) and compares it to `model.total_bytes()`.
It returns:

```cpp
struct cluster_memory_fits_result {
    bool   fits;
    double required_gb;
    double available_gb;
    double missing_gb;
};
```

## Planner algorithm

`dist_plan_layers_memory_aware()`:

1. Compute per-layer cost = `layers[i].weight_bytes + overhead_share`, where
   overhead share is `(kv + compute + scratch) / n_layer`.
2. Sort nodes by benchmark score descending.
3. For each node, consume as many remaining layers as its primary budget
   allows, one layer at a time.
4. Fail if the final cursor does not reach `n_layer`.

The result is a contiguous, gapless layer assignment.  Nodes that cannot hold
a single layer are omitted.  If a model fits, every assigned node runs within
its own GPU or CPU budget.

## Node agent

On startup the node agent now probes memory, CPU and system info with
`dist_probe_node_memory()`, `dist_probe_node_cpu()` and
`dist_probe_node_system()` and sends them in the `/register` payload:

```json
{
  "node_id": "...",
  "memory": {
    "total_ram": 34359738368,
    "free_ram": 26843545600,
    "total_vram": 12884901888,
    "free_vram": 10737418240,
    "has_gpu": true
  },
  "hardware": {
    "backend": "Metal",
    "gpu_name": "Apple M3 Max",
    "cpu_name": "Apple M3 Max",
    "logical_cores": 16,
    "physical_cores": 16
  },
  "system": { "os": "macos", "arch": "arm64" }
}
```

`GET /capabilities` returns the same profiling fields.

## Orchestrator endpoints

### `GET /nodes`

Returns all registered nodes. Use `?format=brief` for a compact list:

```json
{
  "nodes": [
    { "node": "A", "score": 12.5, "backend": "CUDA",
      "free_ram_gb": 32.0, "free_vram_gb": 24.0 }
  ]
}
```

Without `format=brief` the full `memory`, `hardware`, `system` and
`performance` objects are included.

### `GET /capacity`

Aggregates total RAM and total VRAM across all online nodes:

```json
{ "cluster": { "ram_gb": 96.0, "vram_gb": 48.0 } }
```

### `POST /planner/simulate`

Estimate and plan without creating a session.  Returns whether the model
fits and, if it does, the layer layout:

```json
{
  "fits": true,
  "required_gb": 58.0,
  "available_gb": 72.0,
  "missing_gb": 0.0,
  "model": "qwen3-70b",
  "memory": {
    "weights_gb": 42.0,
    "kv_gb": 14.0,
    "compute_gb": 1.5,
    "scratch_gb": 0.5,
    "total_gb": 58.0
  },
  "layout": [
    { "node": "cuda-a", "layers": [0, 28],  "device_hint": "gpu" },
    { "node": "metal-b", "layers": [28, 62], "device_hint": "gpu" },
    { "node": "cpu-c",   "layers": [62, 80], "device_hint": "cpu" }
  ]
}
```

If it does not fit, `fits` is false, `error` explains why, and `layout` is
empty.

### `POST /session/create`

The session endpoint now runs the same estimator/feasibility/planner
pipeline.  If the model does not fit, it returns HTTP 503 with the detailed
fit result, so the caller can surface the deficit in a UI.

On success the response also contains a `memory` block with
`required_gb`, `weights_gb`, `kv_gb`, `compute_gb` and `scratch_gb`.

### `GET /planner/explain`

Returns a human-readable planning report without creating a session.
Useful for a Web UI preview:

```bash
curl "http://orchestrator:8080/planner/explain?model=llama-3.2-1b&n_ctx=4096"
```

Response:

```json
{
  "model": "llama-3.2-1b",
  "fits": true,
  "required_gb": 2.4,
  "available_gb": 8.0,
  "memory": { "weights_gb": 1.9, "kv_gb": 0.2, ... },
  "nodes": [
    {
      "node_id": "cuda-a",
      "backend": "CUDA",
      "free_ram_gb": 32,
      "free_vram_gb": 12,
      "primary_budget_gb": 12,
      "primary_budget_kind": "VRAM",
      "assigned_layers_start": 0,
      "assigned_layers_end": 16,
      "assigned_layer_count": 16,
      "assigned_device": "gpu"
    }
  ],
  "layout": [ { "node": "cuda-a", "layers": [0, 16], "device_hint": "gpu" } ],
  "explanation": "Model llama-3.2-1b requires 2.4 GB ..."
}
```

## Planner report

The orchestrator prints a human-readable planner report to stderr on every
successful session create:

```
Planner Report
Model: qwen3-70b
Weights: 42.0 GB
Estimated KV: 14.0 GB
Compute: 2.0 GB
Scratch: 1.0 GB
Required: 59.0 GB
Cluster:
  Node cuda-a  CUDA/GPU  free VRAM 12.0 GB  free RAM 32.0 GB
  Node metal-b Metal/GPU free VRAM 18.0 GB  free RAM 32.0 GB
  Node cpu-c   CPU/CPU   free VRAM 0.0 GB   free RAM 24.0 GB
Result Fits: YES
Layer layout
  Node cuda-a 0-24
  Node metal-b 24-58
  Node cpu-c 58-80
```

This is the basis for a future `GET /planner/explain` Web UI view.

## Cross-platform probing

`tools/distributed/dist_common.cpp` now contains platform-specific probes:

- **Linux**: `sysinfo()` with `/proc/meminfo` fallback; CPU info via
  `/proc/cpuinfo`; GPU/VRAM via `ggml_backend_dev_memory()`.
- **macOS**: `sysctl` for RAM, CPU name, core count and L3 cache; `host_statistics64`
  for free RAM.
- **Windows**: `GlobalMemoryStatusEx()` for RAM; `__cpuid` for CPU brand;
  GPU/VRAM via `ggml_backend_dev_memory()`.

The legacy `dist_probe_memory()` and `dist_probe_capabilities()` remain for
backwards compatibility with Task 5/6 code.

## Tests

New unit tests are built when `LLAMA_BUILD_TESTS=ON`:

```bash
make -C build test-memory-estimator test-capacity-check \
  test-memory-aware-planner test-planner-no-overlap \
  test-planner-low-memory test-capacity-api test-layer-planner

./build/bin/test-memory-estimator
./build/bin/test-capacity-check
./build/bin/test-memory-aware-planner
./build/bin/test-planner-no-overlap
./build/bin/test-planner-low-memory
./build/bin/test-capacity-api
./build/bin/test-layer-planner
```

Notes:

- `test-memory-estimator` validates the catalog fallback; to validate against
  a real GGUF set `MODEL=/path/to/model.gguf`.
- `test-capacity-check` checks fit/missing logic with independent budgets.
- `test-memory-aware-planner` and `test-planner-*` verify contiguous, gapless,
  budget-respecting layouts on synthetic models.
- `test-capacity-api` validates the `/capacity` response shape in a local
  HTTP server.
- `test-layer-planner` confirms the original Task 6 score-only planner
  still works.

## Migration notes

- Old clients that send only `memory_total_mb`/`memory_free_mb` continue to
  work; missing byte-level fields simply zero out.
- Old clients that call `POST /session/create` receive the same top-level
  `session_id`, `layout` and `pipeline` fields; the new `memory` block is
  additive.
- Existing E2E tests should not need changes unless they assert exact
  node-memory values from `/nodes`.
