# Model Compatibility Matrix (Task 9.8.3)

Automated results from `test-verification-multi-model` and cluster E2E runs.
Update after running tests with local GGUF files (`MODEL` or `VERIFY_MODELS` env).

| Model | Descriptor | Layer store | Materialize | Generate |
|-------|------------|-------------|-------------|----------|
| TinyLlama | PASS | PASS | PASS | PASS (cluster) |
| Llama 3.2 | PASS | PASS | PASS | PASS (cluster) |
| Qwen 2.5 | PASS | PASS* | PASS* | pending cluster |
| Gemma | PASS | — | — | — |
| SmolLM | PASS (llama plugin) | — | — | — |

\* Qwen layer store/materialize fixed in 9.8.3: `output_head` and `output_norm` are
separate per-tensor blobs instead of a merged `layer -2` range.

## How to refresh

```bash
cd llama.cpp
cmake -B build-test -DLLAMA_BUILD_TESTS=ON
cmake --build build-test --target test-verification-multi-model test-qwen-materialization -j
VERIFY_MODELS="~/models/tinyllama.gguf,~/models/qwen2.5.gguf" ./build-test/bin/test-verification-multi-model
```

Cluster generate: register → manifest → layout → sync → generate per model.
