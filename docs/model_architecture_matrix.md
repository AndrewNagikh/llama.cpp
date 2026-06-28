# Model Architecture Matrix (Task 9.8.3)

Research summary for architectures supported by llama.cpp. Used to design semantic
blobs and plugins. Only **llama**, **qwen**, and **gemma** have dedicated plugins
in this task; others fall through to the llama plugin until a specific plugin is added.

| Family | GGUF `general.architecture` | Embedding | LM head | Output norm | Tied emb | Notes |
|--------|----------------------------|-----------|---------|-------------|----------|-------|
| TinyLlama | `llama` | `token_embd.weight` | (tied) | `output_norm.weight` | yes | llama plugin |
| Llama 3.x | `llama` | `token_embd.weight` | `output.weight` or tied | `output_norm.weight` | often yes | 3.2 1B tied |
| Qwen 2.x | `qwen2` | `token_embd.weight` | `output.weight` | `output_norm.weight` | no | head at start of tensor data |
| Qwen 3 | `qwen3` | same pattern | separate head | `output_norm.weight` | no | qwen plugin |
| Gemma | `gemma` / `gemma2` | `token_embd.weight` | varies | `output_norm.weight` | often yes | gemma plugin |
| Phi | `phi3` / `phi` | `token_embd.weight` | varies | `output_norm.weight` | varies | llama plugin fallback |
| SmolLM | `llama` | `token_embd.weight` | varies | `output_norm.weight` | varies | llama plugin fallback |
| DeepSeek | `deepseek` / `deepseek2` | `token_embd.weight` | MoE-specific | `output_norm.weight` | varies | MoE: router + experts in layer blobs |
| Mistral | `llama` | `token_embd.weight` | `output.weight` | `output_norm.weight` | no | llama plugin fallback |

## Special tensor roles

| Role | Typical tensor names |
|------|---------------------|
| embedding | `token_embd.weight` |
| output_head | `output.weight` |
| output_norm | `output_norm.weight` |
| input_norm | `norm.weight`, `token_embd_norm.weight` |
| rotary | `rope_freqs.*`, `rot_embd.*` |
| router | `ffn_gate_inp.weight`, `*.gate.*` |
| expert | `*exps*`, `*expert*` |

## Task 9.9 — semantic runtime descriptor

Plugins now build descriptors via `tensor_graph` (classify all tensors → semantic blobs).
Blob deploy targets drive install and coverage:

| Blob role | Deploy target |
|-----------|---------------|
| embedding, input_norm | entry node |
| output_head, output_norm | final node |
| rope (Llama 3+) | all nodes |
| layer:N | layout node for layer N |

Tied embeddings: `output_head` storage-aliases `embedding`; embedding deploy becomes `all_nodes`.
Coverage READY requires layer indices **and** semantic blobs on correct layout nodes.

DeepSeek / Qwen-MoE add per-layer `router` and `expert` tensors inside `layer:N`
blobs. `architecture_descriptor.is_moe` is set when architecture name contains `moe`.
