# Task 6: Dynamic Layer Planner

## Status

**DONE**

## Goal

Replace hardcoded layer layout `[0,5) [5,10) [10,16)` with automatic proportional assignment based on node performance scores.

## Architecture

```
POST /session/create
    |
    +--> collect registered nodes (score, hardware)
    |
    +--> dist_plan_layers(n_layers, nodes)
    |
    +--> configure agents with dynamic layer ranges
    |
    +--> return layout in session response
```

## Node Agent Changes

### CLI

```bash
node_agent --score 100 ...
```

Default score: `1.0`

### GET /capabilities

```json
{
  "node_id": "node-a",
  "hardware": {
    "cpu_threads": 16,
    "ram_gb": 32,
    "gpu_name": "none",
    "gpu_vram_gb": 0
  },
  "performance": {
    "score": 100.0
  }
}
```

### POST /register

Now includes `score`, `hardware`, `performance.score`.

## Orchestrator Changes

### GET /nodes

Returns `score`, `hardware`, `last_seen` per node.

### POST /session/create

Response includes dynamic `layout`:

```json
{
  "session_id": "sess-...",
  "layout": [
    {"node":"node-a","start":0,"end":9,"score":100,"role":"entry"},
    {"node":"node-b","start":9,"end":14,"score":50,"role":"middle"},
    {"node":"node-c","start":14,"end":16,"score":25,"role":"final"}
  ],
  "pipeline": [...]
}
```

### POST /configure

Now includes `session_id` and dynamic `layer_start` / `layer_end`.

## Layer Planner

Files: `tools/distributed/layer_planner.h`, `layer_planner.cpp`

```cpp
std::vector<dist_layer_assignment> dist_plan_layers(
    int n_layers,
    const std::vector<dist_planner_node> & nodes);
```

### Algorithm v1

1. Sort nodes by score descending
2. Compute proportional layer counts (largest remainder method)
3. Assign contiguous ranges `[0, n_layers)` with no gaps/overlaps

### Examples

| Nodes | Scores | n_layers | Result |
|-------|--------|----------|--------|
| 1 | 100 | 16 | [0,16) |
| 2 | 100, 100 | 16 | [0,8), [8,16) |
| 3 | 100, 50, 25 | 16 | [0,9), [9,14), [14,16) |
| 3 | 100, 50, 25 | 80 | [0,46), [46,69), [69,80) |

## Run (multi-machine)

```bash
# Linux orchestrator + node-a (score 100)
./orchestrator --model model.gguf --listen 0.0.0.0:9000
./node_agent --model model.gguf --listen 0.0.0.0:9001 \
  --advertise-host 192.0.2.10 --orchestrator http://127.0.0.1:9000 \
  --node-id node-a --score 100

# Mac node-b (score 50)
./node_agent ... --node-id node-b --score 50 --listen 0.0.0.0:9002 \
  --advertise-host 192.0.2.12 --orchestrator http://192.0.2.10:9000

# Mac node-c (score 25)
./node_agent ... --node-id node-c --score 25 --listen 0.0.0.0:9003 \
  --advertise-host 192.0.2.12 --orchestrator http://192.0.2.10:9000
```

```bash
curl -s http://192.0.2.10:9000/session/create \
  -H 'Content-Type: application/json' -d '{"model":"llama-3.2-1b"}' | jq '.layout'
```

## Tests

```bash
./build/bin/test-layer-planner
./build/bin/test-orchestrator-dynamic-layout MODEL.gguf
./build/bin/test-orchestrator-3node MODEL.gguf   # regression
```

## Not implemented

- Auto benchmark for score
- Fault tolerance / node migration
- Dynamic rebalancing mid-session
