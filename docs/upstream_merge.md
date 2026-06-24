# Upstream Merge Workflow

How to pull changes from [ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp) into the `feature/distributed-runtime` branch.

See also: [upstream_strategy.md](upstream_strategy.md) for what we changed and why.

---

## One-time setup

```bash
git remote add upstream https://github.com/ggml-org/llama.cpp.git
git fetch upstream
```

---

## Update from upstream

```bash
git checkout feature/distributed-runtime
git fetch upstream

# Inspect incoming changes
git log --oneline HEAD..upstream/master | head -20

# Merge (or rebase if you prefer linear history)
git merge upstream/master
```

If `fatal: refusing to merge unrelated histories` — our fork may need a merge-base repair. Use the documented fork point:

```bash
git merge bddfd2b11   # last known common ancestor
# then merge upstream/master in a second step
```

---

## Files that usually conflict

Resolve these **first** — they contain the core distributed patch:

| Priority | File | Why |
|----------|------|-----|
| 1 | `src/models/llama.cpp` | Partial layer loop in forward pass |
| 2 | `src/llama-context.cpp` | Hidden state + layer range + decode validation |
| 3 | `src/llama-graph.cpp` | `build_inp_hidden()` |
| 4 | `src/llama-graph.h` | Graph context fields |
| 5 | `src/llama-context.h` | Member variables + methods |
| 6 | `src/llama-cparams.h` | `layer_start` / `layer_end` fields |
| 7 | `include/llama-distributed.h` | Keep our file (upstream won't have it) |
| 8 | `common/common.{h,cpp}`, `common/arg.cpp` | Optional CLI flags — may conflict |

**Usually clean (ours only):**

- `tools/distributed/**`
- `docs/task*.md`, `docs/upstream_*.md`
- `tests/test-partial-forward.cpp`, `test-injection-proof.cpp`
- `tests/test-autoregressive-split*.cpp`

**Review carefully:**

- `tools/server/server-context.cpp` — may have both upstream and fork changes
- `tests/CMakeLists.txt`, `tools/CMakeLists.txt` — additive `LLAMA_DISTRIBUTED` blocks

---

## Merge checklist

After resolving conflicts:

### 1. Configure both build modes

```bash
cmake -B build-off -DLLAMA_DISTRIBUTED=OFF -DLLAMA_BUILD_TESTS=OFF
cmake --build build-off -j

cmake -B build -DLLAMA_DISTRIBUTED=ON -DLLAMA_BUILD_TESTS=ON
cmake --build build -j
```

### 2. Core API tests

```bash
./build/bin/test-partial-forward
./build/bin/test-injection-proof
```

### 3. Transport + pipeline tests

```bash
./build/bin/test-split-tcp
./build/bin/test-autoregressive-split
./build/bin/test-autoregressive-split-3node
```

### 4. Distributed runtime tests (require model path)

```bash
./build/bin/test-layer-planner
./build/bin/test-orchestrator-3node path/to/model.gguf
./build/bin/test-orchestrator-dynamic-layout path/to/model.gguf
./build/bin/test-node-benchmark path/to/model.gguf
./build/bin/test-registration-benchmark path/to/model.gguf
```

### 5. Manual smoke test

```bash
# Terminal 1
./build/bin/orchestrator --model model.gguf --listen 0.0.0.0:9000

# Terminal 2
./build/bin/node_agent --model model.gguf --listen 0.0.0.0:9001 \
  --orchestrator http://127.0.0.1:9000 --node-id node-a

curl -s http://127.0.0.1:9000/session/create \
  -H 'Content-Type: application/json' \
  -d '{"model":"llama-3.2-1b"}' | jq .layout
```

---

## Conflict resolution tips

### `src/models/llama.cpp`

Keep our block:

```cpp
const int32_t layer_start = cparams.layer_start;
// ...
if (layer_start > 0) {
    inpL = build_inp_hidden();
}
for (int il = layer_start; il < layer_end; ++il) {
```

Merge upstream changes to the rest of the forward function around this block.

### `src/llama-context.cpp`

Preserve:

- `set_layer_range()`, `set_hidden_state()`, `clear_hidden_state()`
- Hidden-state validation in the decode path
- C wrappers at file bottom (`llama_set_layer_range`, etc.)

### `src/llama-ext.h`

Upstream may add new staging APIs. **Do not** re-add distributed functions here — they belong in `include/llama-distributed.h` only.

---

## When to escalate

- Upstream refactors `llama_context` or graph build → re-apply patch using `docs/task1_partial_forward.md` and `docs/task2_hidden_injection.md`
- Upstream adds native pipeline / RPC → evaluate replacing our TCP workers
- Merge takes >1 day → consider extracting core patch into a single `patches/distributed-core.patch` file
