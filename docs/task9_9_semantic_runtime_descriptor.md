# Task 9.9 — Semantic Runtime Descriptor & Universal Architecture Support

## Goal

Remove all model-family knowledge from Distributed Runtime infrastructure.

After this task:

- supporting a new architecture requires **only** a new Architecture Plugin;
- Materializer has no `include_embedding` / `include_output` and no family `if`s;
- Install Planner builds **blob-only** operations (no negative layer ids, no ad-hoc byte-range layer ops);
- Worker Builder and Coverage consume a single **Semantic Runtime Descriptor**;
- most modern GGUF models run without changing orchestrator, sync, or node agent plumbing.

## Why this is necessary

Multi-node generate failures (Task 9.8 cluster investigation) showed that bugs were not in individual models but in **runtime assumptions**:

| Assumption (wrong) | Reality |
|--------------------|---------|
| output tensors are adjacent in GGUF | Qwen: `output.weight` at tensor-data start, `output_norm` at end |
| embedding only on ENTRY | Llama 3 tied head: final stage needs embedding alias |
| output only on FINAL | correct, but blob must be on **layout final node**, not “somewhere in cluster” |
| rope only on entry | Llama 3+: `rope_freqs.weight` on **every** pipeline stage |
| coverage READY = can generate | READY only checked layer indices, not blob topology or worker buildability |

These assumptions lived in Materializer flags, hybrid install planning, and layer-index coverage — not only in plugins.

## Target architecture

Four layers, strict separation:

```
GGUF manifest
      ↓
Architecture Plugin   (research + minimal overrides)
      ↓
Semantic Runtime Descriptor   (single source of truth)
      ↓
Distributed Runtime   (family-agnostic)
```

**Distributed Runtime must not import or branch on `llama`, `qwen`, `gemma`, etc.**

---

## Stage 1 — Architecture Plugins

**Location:** `tools/distributed/architecture/plugins/`

Minimum plugins: `llama_plugin`, `qwen_plugin`, `gemma_plugin`.

Plugin responsibilities:

- read `general.architecture` and GGUF metadata;
- analyze the **full** tensor list;
- emit a Semantic Runtime Descriptor;
- **never** download, materialize, or run inference.

### Mandatory: research-first plugins

Each plugin is primarily a **research module**, not a hardcoded name list.

Algorithm (in order):

1. Read `general.architecture` and related metadata keys from manifest.
2. Classify **every** tensor by structure (prefix, layer index, role heuristics).
3. Build a dependency graph:
   - layer-attached vs global;
   - entry-only vs final-only vs all-stages;
   - tied / aliased storage (same bytes as another blob).
4. Emit semantic blobs and worker requirements from the graph.
5. Apply **minimal** family-specific overrides only when automatic inference is ambiguous.

Goal: Qwen 4 / Gemma 4 with small tensor naming changes should mostly work without infrastructure changes.

**Current state (9.8.3):** plugins exist but use prefix matching + hand-written blob rules. **Gap:** tensor graph analyzer not implemented.

---

## Stage 2 — Semantic Runtime Descriptor

Replace ad-hoc `architecture_descriptor` + scattered plans with one runtime struct:

```cpp
struct worker_descriptor {
    worker_role              role;
    std::vector<std::string> required_blob_ids;
    std::vector<int32_t>     required_layer_indices;  // empty = range from session layout
};

struct semantic_runtime_descriptor {
    architecture_descriptor          architecture;  // family, tied_embeddings, is_moe, …
    std::vector<semantic_blob>       blobs;
    std::vector<worker_descriptor>   workers;
};
```

The descriptor is the **only** input to Install Planner, Materializer, Coverage, and Worker Builder.

**Current state:** `architecture_descriptor` + `worker_requirement` partially cover this. **Gap:** no unified `semantic_runtime_descriptor`; workers not materialized from explicit `worker_descriptor`.

---

## Stage 3 — Semantic Blob

Eliminate special layer ids `-1`, `-2`, `-3` in storage and APIs.

Blob is a first-class install/materialize unit:

| blob_id | tensors (example) |
|---------|-------------------|
| `embedding` | `token_embd.weight` |
| `output_head` | `output.weight` |
| `output_norm` | `output_norm.weight` |
| `rope` | `rope_freqs.weight` |
| `layer_17` | `blk.17.*` |

Blob stores:

- tensor names + manifest offsets (for download only — not stored on blob after install);
- checksum policy per tensor;
- aggregate size;
- deploy target (`entry_node`, `final_node`, `all_nodes`);
- optional storage alias (`output_head` → `embedding` when tied).

Blob does **not** define a contiguous GGUF byte range for install (per-tensor downloads).

**Current state:** blob ids in `blobs/{id}/{tensor}.bin`; layer blobs use `layer:N`. **Gap:** `layer_special.h` still has `-1`/`-2`; install planner still uses layer byte ranges for transformer layers.

---

## Stage 4 — Runtime Requirements

```cpp
struct runtime_requirement {
    worker_role              role;
    std::vector<std::string> required_blob_ids;
    std::vector<int32_t>     required_layer_indices;
};
```

Examples (from descriptor, not from code):

| Role | typical blobs |
|------|----------------|
| ENTRY | `embedding`, `rope?`, `input_norm?`, `layer_*` |
| MIDDLE | `rope?`, `layer_*` |
| FINAL | `output_norm`, `output_head`, `rope?`, `embedding?` (if tied), `layer_*` |
| ALL / metadata | `metadata` (header KV + tokenizer tables) |

**Current state:** `worker_requirement` + `materialize_plan_for_worker`. **Gap:** materializer still driven by boolean flags at call sites.

---

## Stage 5 — Blob Builder

`architecture/blob_builder.*` — builds semantic blobs from manifest tensor list.

- knows tensor → blob grouping rules;
- used by plugins and tests;
- Install Planner does **not** embed grouping logic.

**Current state:** implemented. **Gap:** move remaining grouping from plugins into shared graph builder.

---

## Stage 6 — Materializer refactoring

**Remove:**

```cpp
layer_store_materialize_gguf(..., include_embedding, include_output)
```

**Replace with:**

```cpp
materialize_worker_gguf(store, manifest, worker_descriptor, layer_range, out_path);
```

Materializer:

1. loads `required_blob_ids` from descriptor;
2. loads layer blobs for `[layer_start, layer_end)` from session layout;
3. assembles partial GGUF from `metadata.bin` + blob tensors;
4. **no** architecture conditionals.

**Files to change:**

- `node_agent/layer_store/layer_gguf_assembler.*`
- `node_agent.cpp` (`materialize_worker_gguf`)
- `verification/worker_tensor_plan.*`, `worker_verify.*`

---

## Stage 7 — Install Planner refactoring

Operations are **blob-centric only**:

```json
{ "action": "DOWNLOAD", "node": "node-c", "blob_id": "output_head", "tensor_name": "output.weight" }
{ "action": "DOWNLOAD", "node": "node-b", "blob_id": "layer_0", ... }
{ "action": "DELETE",   "node": "node-a", "blob_id": "output_head", "tensor_name": "output.weight" }
```

No negative `layer_index` in public install API. Transformer layers use `layer_N` blob ids, not `manifest_layer_byte_range` merged ranges.

Reconciliation (misplaced blobs): compare actual blob locations vs descriptor deploy targets for entry/final/all_nodes.

**Files to change:**

- `orchestrator/install_planner/install_planner.cpp`
- `architecture/install_planning.cpp`

---

## Stage 8 — Coverage refactoring

Three levels:

| Level | Question |
|-------|----------|
| **Storage coverage** | Is blob `B` present on node `N` with valid checksum? |
| **Runtime coverage** | Given layout + descriptor, can each worker role be materialized on its node? |
| **Session coverage** | Can orchestrator configure the full pipeline (peers, ports, blobs local)? |

`coverage.state == READY` iff **runtime coverage** passes for every node in desired layout — not merely “all layer indices exist somewhere in the cluster”.

**Files to change:**

- `orchestrator/coverage/coverage.cpp`
- `orchestrator/coverage/coverage.h`
- cluster test / API responses

---

## Stage 9 — Architecture research

Extend `docs/model_architecture_matrix.md` for:

TinyLlama, Llama 3.x, Qwen 2.x, Qwen 3, Gemma 2/3, Phi 3/4, SmolLM, Mistral, DeepSeek, Mixtral.

Per family document:

- global tensors;
- layer tensor patterns;
- tied embeddings / lm_head;
- rope placement;
- MoE router/experts;
- shared tensors;
- known GGUF layout quirks (e.g. Qwen head at low offset).

Feed research into plugin graph rules and regression matrix.

---

## Stage 10 — Compatibility matrix

Automated pipeline per model:

```
Register → Manifest → Descriptor → Install → Materialize → Generate
```

Track per model:

| Model | Descriptor | Materialize | Single-node | Multi-node |
|-------|------------|-------------|-------------|------------|
| TinyLlama | | | | |
| Llama 3.2 | | | | |
| Qwen 2.5 | | | | |
| Gemma 3 | | | | |

Script: extend `scripts/cluster_multiarch_test.py` + local `test-verification-multi-model`.

Failures must pinpoint stage (descriptor vs materialize vs session vs generate).

---

## Stage 11 — Tests

### Unit

- `test-architecture-detector` ✓ (exists)
- `test-plugin-llama` ✓ (`test-llama-plugin.cpp`)
- `test-plugin-qwen` ✓
- `test-plugin-gemma` ✓
- `test-semantic-blob` ✓
- `test-runtime-descriptor` — **new**
- `test-worker-builder` — **new**

### Integration

- `test-qwen-runtime` — **new**
- `test-gemma-runtime` — **new**
- `test-tinyllama-runtime` — **new**
- `test-cross-architecture-materialization` — **new**

### E2E

Multi-arch cluster test: TinyLlama → Llama → Qwen → Gemma, each through session generate with non-repetitive output.

---

## Constraints (out of scope for 9.9)

Do **not** change:

- distributed prefill algorithm;
- distributed KV cache;
- TCP / split_gen wire protocol;
- layer planner memory algorithm.

Only model description, install/materialize, coverage, and worker assembly.

---

## Acceptance criteria

Task is **done** when:

- [ ] Runtime, Materializer, Install Planner contain **no** `llama`/`qwen`/`gemma` conditionals;
- [ ] Worker Builder uses `semantic_runtime_descriptor` only;
- [ ] Coverage READY implies worker buildable on layout nodes (blobs + layers);
- [ ] No `-1`/`-2`/`-3` layer ids in install/sync/storage paths;
- [ ] TinyLlama, Llama 3.2, Qwen 2.5, Gemma 3 pass full pipeline **single-node and multi-node**;
- [ ] New architecture = new plugin only (research graph + minimal overrides);
- [ ] Compatibility matrix runs in CI or documented homelab script.

---

## Relationship to Task 9.8.3

Task 9.8.3 delivered the **foundation** (~50% of 9.9):

| 9.8.3 delivered | 9.9 completes |
|-----------------|---------------|
| Plugin interface + 3 plugins | Research-first tensor graph in plugins |
| `semantic_blob`, blob store paths | Remove `layer_special` legacy |
| `add_semantic_blob_downloads` | Blob-only layer install |
| `descriptor_materialize` | Flag-free materializer API |
| Family logic out of orchestrator | Runtime/session coverage |

---

## Recommended implementation order

1. **Stage 2 + 4** — `semantic_runtime_descriptor`, `worker_descriptor`, builder from existing descriptor.
2. **Stage 6** — materializer API switch (highest impact on multi-node generate).
3. **Stage 7** — layer downloads as `layer_N` blobs; remove byte-range layer ops.
4. **Stage 8** — runtime coverage (fixes false READY).
5. **Stage 1 research refactor** — tensor graph in plugins; shrink hand-coded rules.
6. **Stage 10–11** — matrix + tests; fix remaining multi-node Qwen/TinyLlama issues.

---

## Files reference (migration map)

| Component | Current | Target |
|-----------|---------|--------|
| Descriptor | `architecture/architecture_descriptor.h` | `architecture/semantic_runtime_descriptor.h` |
| Plugins | `architecture/plugins/*_plugin.cpp` | + `architecture/tensor_graph.*` |
| Materialize | `layer_gguf_assembler.cpp` (flags) | `worker_builder.cpp` (descriptor) |
| Install | `install_planner.cpp` (hybrid) | blob-only via `install_planning.cpp` |
| Coverage | `coverage/coverage.cpp` (layers) | storage + runtime + session |
| Legacy | `layer_special.h`, `architecture_descriptor/` | delete after migration |
