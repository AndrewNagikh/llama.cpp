# Task 9.8.2 — Architecture Descriptor & Semantic Tensor Roles

## Goal

Remove model-family conditionals (`if (llama)`, `if (tied_embeddings)`) from Materializer,
Install Planner, Verification, and Worker Tensor Plan. All architecture rules live in a
single **Architecture Descriptor** built from GGUF metadata and tensor inventory.

```
GGUF Manifest
     ↓
Architecture Descriptor  (descriptor_builder)
     ↓
Semantic Tensor Roles
     ↓
Worker Tensor Plan / Install Plan / Materialization / Verification
```

## How llama.cpp determines architecture

When a GGUF file is loaded, llama.cpp uses a combination of **metadata keys** and **tensor
naming conventions** (standardized at conversion time):

| Source | Examples | Used for |
|--------|----------|----------|
| `general.architecture` | `llama`, `qwen2`, `gemma3`, `phi3`, `deepseek2` | Family dispatch, metadata key prefix |
| `{arch}.block_count` | layer count | Layout, layer split |
| `{arch}.context_length` | max context | Runtime (not changed in 9.8.2) |
| `{arch}.vocab_size` | vocabulary size | Manifest |
| `{arch}.embedding_length` | hidden size | Manifest |
| Tensor inventory | presence/absence of `output.weight` | **Tied embeddings** inference |
| Standard tensor names | `token_embd.weight`, `blk.N.*` | Role classification |

**Tied embeddings** are inferred the same way as `llama-quant.cpp`: assume tied until
`output.weight` (OUTPUT category) is seen. No separate metadata flag is required.

MoE architectures are detected via `general.architecture` containing `moe` (e.g.
`qwen2moe`, `mixtral`, `deepseek2`).

## Supported architecture survey

All listed families use llama.cpp-standard GGUF tensor names after conversion (`token_embd.weight`,
`blk.N.attn_*`, `blk.N.ffn_*`, `output_norm.weight`).

| Architecture | GGUF `general.architecture` | Tied embeddings | Output norm | Notable extras | Attention notes |
|--------------|----------------------------|-----------------|-------------|----------------|-----------------|
| Llama 2 | `llama` | Usually tied | `output_norm.weight` | — | GQA in larger variants |
| Llama 3 / 3.1 / 3.2 | `llama` | Usually tied | `output_norm.weight` | RoPE in graph | GQA |
| Mistral | `llama` (sliding window in metadata) | Tied | `output_norm.weight` | `llama.attention.sliding_window` | SWA |
| Mixtral | `llama` + MoE tensors | Tied | `output_norm.weight` | `blk.N.ffn_gate_inp`, `*_exps` | MoE FFN |
| Gemma 2 | `gemma2` | Often tied | `output_norm.weight` | Extra RMS norms per block | — |
| Gemma 3 | `gemma3` | Often tied | `output_norm.weight` | Gemma3-specific metadata | — |
| Qwen 2 / 2.5 | `qwen2` | Separate `output.weight` typical | `output_norm.weight` | QKV bias optional | GQA |
| Qwen 3 | `qwen3` | Separate head typical | `output_norm.weight` | — | — |
| Qwen 2/3 MoE | `qwen2moe`, `qwen3moe` | Varies | `output_norm.weight` | Router + expert tensors | MoE |
| Phi 3 / 4 | `phi3` | Varies by checkpoint | `output_norm.weight` | — | — |
| DeepSeek / V2 / V3 | `deepseek`, `deepseek2` | Varies | `output_norm.weight` | MLA tensors in V2+ | MLA / MoE in V2+ |
| SmolLM / SmolLM2 | `smollm3` (SmolLM3) / llama-like | Often tied | `output_norm.weight` | Compact blocks | — |
| TinyLlama | `llama` | Tied | `output_norm.weight` | Same as Llama family | — |

### Architecture-specific tensors (MoE)

| Pattern | Semantic role | Worker need |
|---------|---------------|-------------|
| `blk.N.ffn_gate_inp.weight` | `router` | Middle (layer range) |
| `blk.N.ffn_*_exps.weight` | `expert` | Middle |
| `blk.N.ffn_*_shared_*` | `shared_expert` | Middle |

MoE full pipeline support is **out of scope** for runtime in 9.8.2; roles are classified so
future work can extend the descriptor without renaming tensors.

## Common patterns

Every dense decoder-only transformer shares:

1. **Embedding** — `token_embd.weight` (ENTRY worker)
2. **Transformer blocks** — `blk.N.*` (MIDDLE / layer-range workers)
3. **Final norm** — `output_norm.weight` (FINAL worker)
4. **Output projection** — `output.weight` OR tied embedding (FINAL worker)

Variations:

- **Input norm** — `norm.weight`, `token_embd_norm.weight` (ENTRY preamble)
- **Separate vs tied output** — presence of `output.weight`
- **Biases** — `.bias` suffix on some Qwen/Phi checkpoints
- **RoPE** — computed at runtime; optional `rot` tensors in some archs

## Semantic tensor roles

```cpp
enum class tensor_semantic_role {
    unknown,
    embedding,
    output_head,
    output_norm,
    input_norm,
    transformer_layer,
    attention,
    ffn,
    expert,
    router,
    shared_expert,
    rotary,
    bias,
    gate,
    metadata,
    other,
};
```

Mapping from manifest `tensor_role` + tensor name patterns is implemented in
`architecture_descriptor/descriptor_builder.cpp` (`classify_tensor_semantic`).

## Worker requirements

```cpp
enum class worker_deploy_role { entry, middle, final, full };

struct tensor_role_requirement {
    tensor_semantic_role semantic;
    bool required_for_entry;
    bool required_for_middle;
    bool required_for_final;
    bool replicate_to_all_nodes;  // install planner: embedding on all nodes when tied
};
```

Default rules (from descriptor):

| Semantic role | ENTRY | MIDDLE | FINAL | Replicate all nodes |
|---------------|-------|--------|-------|---------------------|
| embedding | yes | — | yes if tied | yes if tied |
| output_head | — | — | yes | — |
| output_norm | — | — | yes | — |
| input_norm | yes | — | — | — |
| transformer_layer | — | layer range | — | — |
| router / expert | — | layer range (MoE) | — | — |

### Examples

**Llama 3.2 (tied):**

```
token_embd.weight → ENTRY + FINAL (via tied output)
output_norm.weight → FINAL
```

**Qwen 2.5 (untied):**

```
token_embd.weight → ENTRY only
output.weight → FINAL
output_norm.weight → FINAL
```

## Architecture Descriptor

```cpp
struct architecture_descriptor {
    std::string architecture;
    bool tied_embeddings;
    bool has_separate_output;
    bool has_output_norm;
    bool is_moe;
    std::map<std::string, std::string> special_tensors;
    std::vector<semantic_tensor_entry> tensors;
    std::vector<tensor_role_requirement> role_requirements;
};
```

Built by:

```cpp
architecture_descriptor build_architecture_descriptor(const model_manifest & manifest);
```

## Descriptor builder rules

1. Read `manifest.architecture` and `special_tensors` from GGUF manifest builder.
2. Set `tied_embeddings = !has_separate_output` where separate output means `lm_head` in specials.
3. Classify each tensor with `classify_tensor_semantic`.
4. Set `is_moe` when architecture string contains `moe`.
5. Emit default `role_requirements` from tied/MoE flags.

**Single place for exceptions:** `descriptor_builder.cpp` only.

## Integration (Task 9.8.2)

| Component | Uses descriptor via |
|-----------|---------------------|
| `manifest_builder` | `classify_tensor_enhanced` |
| `layer_gguf_assembler` | `architecture_materialize_needs_embedding_for_output` |
| `install_planner` | `architecture_should_replicate_embedding` |
| `worker_tensor_plan` | `architecture_tensor_included`, `build_architecture_descriptor` |
| `worker_verify` | `architecture_materialize_needs_embedding_for_output` |

No component outside `architecture_descriptor/` checks `special_tensors.count("lm_head")`.

## Limitations

- Does **not** change runtime inference, prefill, KV cache, or wire protocol.
- MoE models: roles are classified; full distributed MoE execution is not validated.
- Architectures with non-standard tensor names (pre-conversion HF names) are unsupported
  until converted to llama.cpp GGUF conventions.
- `general.architecture` values not present in upstream llama.cpp may need descriptor
  builder updates (one file).

## Tests

### Unit tests

| Test | Purpose |
|------|---------|
| `test-architecture-descriptor` | Build descriptor from synthetic manifests |
| `test-semantic-roles` | Tensor name → semantic role |
| `test-worker-requirements` | ENTRY/FINAL embedding rules |
| `test-tied-embeddings` | FINAL plan includes embedding when tied |
| `test-output-head` | FINAL plan includes `output.weight` when untied |
| `test-qwen-descriptor` | Real Qwen GGUF (skip if absent) |
| `test-gemma-descriptor` | Real Gemma GGUF (skip if absent) |
| `test-deepseek-descriptor` | Real DeepSeek GGUF (skip if absent) |

### Integration / compatibility matrix

`test-verification-multi-model` prints a matrix after running the verification pipeline:

| Model | Manifest | LayerStore | Materialization | RuntimeLoad | HiddenState | Logits | Sampling |
|-------|----------|------------|-----------------|-------------|-------------|--------|----------|
| Llama 3.2 | | | | | | | |
| TinyLlama | | | | | | | |
| Qwen 2.5 | | | | | | | |
| Gemma 3 | | | | | | | |
| SmolLM2 | | | | | | | |
| Phi 3.5 | | | | | | | |

Set `VERIFY_MODELS` to a comma-separated list of GGUF paths for homelab regression.

### Recommended probe order

1. TinyLlama — cheap Llama-family sanity check
2. Qwen 2.5 — non-Llama family
3. Gemma 3 — distinct metadata/norms
4. SmolLM2 — compact modern arch
5. Phi 3.5 — additional family coverage

Optional MoE probes (expected partial failure): Qwen3-30B-A3B, Mixtral — inform
`expert` / `router` role design.

## Adding a new architecture

1. Ensure llama.cpp converts the model to standard GGUF tensor names.
2. If tensor patterns are new, extend `classify_tensor_semantic` in `descriptor_builder.cpp`.
3. If worker rules differ, extend `default_role_requirements` in the same file.
4. Add a descriptor unit test with synthetic manifest or a small GGUF fixture.
5. Run `test-verification-multi-model` with the new GGUF in `VERIFY_MODELS`.

No changes required in Materializer, Install Planner, or Verification.
