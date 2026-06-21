# Task 5: Basic Orchestrator + Node Agent

## Status

**DONE**

## Goal

Replace hardcoded `fork()` / localhost test setup with real networked nodes orchestrated over HTTP.

Three independent machines (or processes) run `node_agent` and participate in a single inference session via the orchestrator.

## Architecture

```
User / curl
    |
    v
Orchestrator (:9000 HTTP)
    |  POST /register, /session/create, /session/generate
    |
    +--> Node A agent (:9001 HTTP) --> split_gen3_a worker
    |         TCP ctrl ----------------^
    |         TCP hidden --> Node B agent --> split_gen3_b
    |                              TCP hidden --> Node C agent --> split_gen3_c
    |                                                    sample token
    +<----------------------- token stream ------------------+
```

## Components

| Binary | Location | Role |
|--------|----------|------|
| `orchestrator` | `tools/distributed/orchestrator.cpp` | HTTP coordinator |
| `node_agent` | `tools/distributed/node_agent.cpp` | HTTP supervisor + worker launcher |
| `split_gen3_{a,b,c}` | `tests/split_gen3_*.cpp` | Pipeline workers (proven Task 4 code) |
| `test-orchestrator-3node` | `tools/distributed/test-orchestrator-3node.cpp` | Integration test |

Separate deploy repo: `distributed-llm/node-agent/` (build script + README).

## Node registration

On startup each agent POSTs to orchestrator `/register`:

```json
{
  "node_id": "node-a",
  "host": "10.0.0.1",
  "port": 9001,
  "n_layer": 16,
  "n_embd": 2048,
  "memory_total_mb": 16384,
  "memory_free_mb": 8192,
  "capabilities": {
    "gpu_backend": "cpu",
    "gpu_memory_mb": 0,
    "cpu_threads": 8,
    "supported_arch": ["llama"]
  }
}
```

## Static pipeline (layout A)

| Stage | node_id | Layers |
|-------|---------|--------|
| Entry | node-a | [0, 5) |
| Middle | node-b | [5, 10) |
| Final | node-c | [10, 16) |

Orchestrator picks first 3 registered nodes sorted by `node_id`.

Configure order: final -> middle -> entry (same as Task 4 startup).

## Session API

### POST /session/create

```json
{ "model": "llama-3.2-1b" }
```

Response includes `session_id` and `pipeline` with assigned ports.

### POST /session/generate

```json
{
  "session_id": "...",
  "prompt": "Tell me a joke",
  "max_tokens": 32
}
```

Returns `{ "tokens": [...], "text": "...", "count": 32 }`.

On pipeline failure: HTTP 503 with `{ "error": "..." }`.

## Network changes (Task 5 + 6 prep)

- `split_tcp_listen_host(host, port)` — bind `0.0.0.0` or specific IP
- `split_tcp_connect_retry(host, port, retries, delay_ms)`
- `split_gen3_*` flags: `--bind`, `--b-host`, `--bc-host`
- TCP socket timeouts (`split_tcp_set_timeouts`) for graceful failure

## Tests

```bash
MODEL=~/models/llama-3.2-1b-instruct-q4_k_m.gguf

# Test 1: orchestrator + 3 agents, compare to split_gen3 baseline
./build/bin/test-orchestrator-3node "$MODEL"

# Test 3: node failure (shutdown middle worker)
./build/bin/test-orchestrator-3node "$MODEL" --test-node-failure
```

### Results (Linux x86_64, CPU)

| Test | Result |
|------|--------|
| 32 tokens vs split_gen3 layout A | **32/32 MATCH** |
| Node B shutdown during generate | **503 graceful error** |

## Manual run (this machine as orchestrator + node)

Terminal 1 — orchestrator:

```bash
./build/bin/orchestrator \
  --model ~/models/llama-3.2-1b-instruct-q4_k_m.gguf \
  --listen 0.0.0.0:9000
```

Terminals 2–4 — agents (same or remote machines):

```bash
./build/bin/node_agent --model ~/models/llama-3.2-1b-instruct-q4_k_m.gguf \
  --listen 0.0.0.0:9001 --orchestrator http://127.0.0.1:9000 --node-id node-a

./build/bin/node_agent ... --listen 0.0.0.0:9002 --node-id node-b
./build/bin/node_agent ... --listen 0.0.0.0:9003 --node-id node-c
```

Generate:

```bash
SID=$(curl -s http://127.0.0.1:9000/session/create \
  -H 'Content-Type: application/json' \
  -d '{"model":"llama-3.2-1b"}' | jq -r .session_id)

curl -s http://127.0.0.1:9000/session/generate \
  -H 'Content-Type: application/json' \
  -d "{\"session_id\":\"$SID\",\"prompt\":\"Tell me a joke\",\"max_tokens\":32}"
```

## Performance notes

- Each node loads the full model GGUF (same as Task 4); memory benefit comes from layer-range execution, not smaller files.
- HTTP orchestration adds negligible overhead vs raw TCP ctrl path.
- Pipeline TCP ports are in 9100+ range (separate from agent HTTP ports).

## Not implemented (future)

- Dynamic balancing, auto layer optimization
- Fault tolerance / reconnect
- KV migration, TLS, authentication

## Build

```bash
cmake -B build -DLLAMA_BUILD_TESTS=ON
cmake --build build --target orchestrator node_agent test-orchestrator-3node \
  split_gen3_a split_gen3_b split_gen3_c -j8
```
