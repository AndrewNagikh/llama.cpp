# Supported Architectures (Distributed Runtime)

См. также реестр в репозитории node-agent: [docs/supported_architectures.md](https://github.com/AndrewNagikh/node-agent/blob/main/docs/supported_architectures.md).

| Architecture | Статус | Partial Forward | Hidden Injection | Layer-first | Generate | Verification |
|--------------|--------|-----------------|------------------|-------------|----------|--------------|
| Llama | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| Qwen | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| Gemma | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| Phi | 🟡 | ✅ | ✅ | ✅ | Sync | ✅ |
| SmolLM | 🟡 | ✅ | ✅ | ✅ | Sync | ✅ |
| DeepSeek-Qwen | 🟡 | Через Qwen | Через Qwen | ✅ | Sync | ✅ |

Plugin registry: `tools/distributed/architecture/descriptor_service.cpp`  
E2E runner: `tools/distributed/docker/run_e2e_generate.py`  
Architecture matrix: `node-agent/config/architecture_matrix.json`
