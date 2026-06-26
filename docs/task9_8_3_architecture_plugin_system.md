# Task 9.8.3 — Architecture Plugin System & Semantic Descriptor

## Goal

Remove model-family knowledge from the distributed runtime. All decisions about
which tensors belong on which worker are driven by an **architecture descriptor**
built by a small architecture plugin.

## Pipeline

```
GGUF manifest
      ↓
Architecture plugin (llama / qwen / gemma)
      ↓
Architecture descriptor
      ↓
Install planner · Materializer · Verification · Sync
```

## Module layout

```
tools/distributed/architecture/
  architecture_plugin.h      — plugin interface
  architecture_descriptor.h  — descriptor + build_architecture_descriptor()
  semantic_blob.h            — per-blob tensor lists (not byte ranges)
  worker_requirement.h       — worker role → required blob ids
  blob_builder.h             — shared blob construction helpers
  tensor_plan.h              — tensor inclusion for worker plans
  install_planning.h         — descriptor-driven install downloads
  plugins/
    llama_plugin.*
    qwen_plugin.*
    gemma_plugin.*
```

Layer store blob storage lives under `node_agent/layer_store/`:

```
{model_id}/blobs/{blob_id}/{tensor_name}.bin
```

## Architecture plugin API

```cpp
class architecture_plugin {
public:
    virtual std::string family() const = 0;
    virtual bool matches(const model_manifest &) const = 0;
    virtual architecture_descriptor build_descriptor(const model_manifest &) const = 0;
};
```

Plugins only analyze manifests. They do not load models, materialize GGUF, or run inference.

Detection order: **qwen → gemma → llama** (llama is the default fallback for
`llama`, `tinyllama`, `mistral`, `phi`, `deepseek`, `smollm`, …).

## Descriptor format

```cpp
struct architecture_descriptor {
    std::string architecture;
    std::string family;
    bool tied_embeddings;
    bool separate_lm_head;
    bool is_moe;
    std::vector<semantic_blob> blobs;
    std::vector<worker_requirement> worker_requirements;
};
```

## Semantic blob format

Each blob is a named group of tensors with manifest offsets — **not** a contiguous
GGUF byte range.

```cpp
struct semantic_blob {
    std::string id;                    // e.g. "embedding", "output_head", "layer:17"
    tensor_semantic_role role;
    std::vector<semantic_tensor_slot> tensors;  // name + offset + size
    blob_deploy_target deploy;         // entry_node | final_node | all_nodes
    bool storage_alias;                // tied output_head → embedding storage
    std::string storage_blob_id;
};
```

### Qwen example

| Blob id       | Tensors              | Deploy    |
|---------------|----------------------|-----------|
| embedding     | token_embd.weight    | entry     |
| output_head   | output.weight        | final     |
| output_norm   | output_norm.weight   | final     |
| layer:N       | blk.N.*              | (planner) |

`output.weight` and `output_norm.weight` are stored and downloaded **separately**
even when their GGUF offsets are far apart.

### Llama tied-embedding example

| Blob id     | Storage                         | Deploy    |
|-------------|---------------------------------|-----------|
| embedding   | token_embd.weight               | all_nodes |
| output_head | alias → embedding (semantic)    | final     |
| output_norm | output_norm.weight              | final     |

## Worker requirement format

```cpp
struct worker_requirement {
    worker_role role;
    int32_t layer_start, layer_end;
    std::vector<std::string> required_blobs;
};
```

| Role   | Typical blobs                          |
|--------|----------------------------------------|
| ENTRY  | embedding, rope/globals, input_norm    |
| MIDDLE | (layer blobs via placement)            |
| FINAL  | output_norm, output_head               |
| FULL   | all of the above                       |

## Install planner

`add_semantic_blob_downloads()` emits one `download_operation` per tensor:

```json
{
  "blob_id": "output_head",
  "tensor_name": "output.weight",
  "tensor_offset": 5950496,
  "tensor_length": 191000000
}
```

No merged `layer -2` byte ranges.

## Materializer

`layer_store_materialize_gguf()` writes transformer layers from layer store, then
calls `materialize_descriptor_tensors()` to place each semantic blob tensor at its
manifest offset.

## Adding a new architecture

1. Implement `architecture_plugin` in `architecture/plugins/`.
2. Register it in `descriptor_service.cpp`.
3. No changes to install planner, materializer, or verification conditionals.

## Tests

| Test | Purpose |
|------|---------|
| test-architecture-detector | plugin selection |
| test-architecture-descriptor | descriptor shape |
| test-semantic-blobs | separate Qwen head/norm offsets |
| test-llama/qwen/gemma-plugin | per-plugin blobs |
| test-qwen-materialization | populate + materialize FINAL worker |

Run with `cmake -DLLAMA_BUILD_TESTS=ON` and `./build-test/bin/test-*`.

## Compatibility

See `docs/model_compatibility.md` and `docs/model_architecture_matrix.md`.
