# Task 7 — Distributed Model Storage & Installation

**Status:** ✅ **COMPLETED**

## Goal

Implement distributed model management where orchestrator manages model catalog and placement across nodes, preparing infrastructure for future layer sharding.

## ✅ Achievements

### 1. Model Catalog System

**New Components:**
- `tools/distributed/model_catalog.h/cpp` - Core model management classes
- `model_catalog` class - Manages available models and installation jobs  
- `model_store` class - Per-node local model storage
- `model_placement` struct - Future layer distribution planning

**Default Catalog:**
```json
{
  "models": [
    {
      "id": "llama-3.2-1b",
      "display_name": "Llama 3.2 1B Instruct Q4_K_M",
      "n_layers": 16,
      "n_embd": 2048, 
      "size_gb": 0.8,
      "source": {
        "repo": "hugging-quants/Llama-3.2-1B-Instruct-Q4_K_M-GGUF",
        "file": "llama-3.2-1b-instruct-q4_k_m.gguf"
      }
    }
  ]
}
```

### 2. New API Endpoints

**Orchestrator APIs:**
- `GET /catalog` - List available models in catalog
- `GET /models` - List installed models across cluster  
- `POST /models/install` - Install model on cluster
- `GET /models/install/{job_id}` - Check installation status

**Node Agent APIs:**
- `GET /models/local` - List locally installed models
- `POST /models/install` - Install model locally (with download)

**Example Workflow:**
```bash
# List available models
curl http://orchestrator:9000/catalog

# Install model 
curl -X POST http://orchestrator:9000/models/install \
  -d '{"model":"llama-3.2-1b"}'

# Check status
curl http://orchestrator:9000/models/install/install-123456
```

### 3. HuggingFace Downloader

**New Tool:** `tools/distributed/download_model.py`
- Downloads GGUF models from HuggingFace repos
- Supports both `huggingface_hub` library and `hf` CLI
- Progress tracking via JSON files
- Storage in `~/.distributed-llm/models/`

**Usage:**
```bash
python3 download_model.py \
  --repo hugging-quants/Llama-3.2-1B-Instruct-Q4_K_M-GGUF \
  --file llama-3.2-1b-instruct-q4_k_m.gguf \
  --output ~/.distributed-llm/models/llama-3.2-1b/model.gguf
```

### 4. Placement Planner

**Layer Distribution Algorithm:**
```cpp
// 16 layers across 3 nodes
node-a: layers [0, 6)   = 6 layers (~307MB)
node-b: layers [6, 11)  = 5 layers (~256MB) 
node-c: layers [11, 16) = 5 layers (~256MB)
```

**Future Ready:** Prepares for Task 8/9 where nodes will store only their assigned layers instead of full GGUF.

### 5. State Persistence

**Orchestrator State:** `state/catalog.json`
- Model catalog with installation jobs
- Recovers after restart

**Node State:** `~/.distributed-llm/models.json` 
- Local model inventory
- Prevents re-downloading existing models

### 6. Comprehensive Tests

**New Tests:**
- `test-model-catalog` - Unit tests for catalog/store classes
- `test-model-install` - Integration test for full workflow

**Test Coverage:**
```
=== Model Catalog Tests ===
✓ model_catalog: job management, model lookup
✓ model_store: local inventory, path management  
✓ placement_planner: layer distribution across nodes
✓ persistence: state save/load

=== Integration Tests === 
✓ catalog API workflow
✓ installation job lifecycle
✓ node model store operations
✓ placement planning for 3-node cluster
✓ state persistence and recovery
```

## 🏗️ Architecture

```
Orchestrator
├─ Node Registry (existing)
├─ Session Manager (existing)  
├─ Layer Planner (existing)
├─ Model Catalog (NEW)
└─ Model Installer (NEW)

Node Agent  
├─ Worker Manager (existing)
├─ Benchmark (existing)
└─ Model Store (NEW)
```

**Clean Separation:**
- **Orchestrator:** Manages catalog, coordinates installation
- **Nodes:** Handle local storage, execute downloads
- **No Docker dependency:** Uses Python scripts + HF tools

## 📊 Current Limitations (By Design)

Per Task 7 requirements, the following are **not implemented:**
- ❌ HuggingFace search/browser
- ❌ GUI for model management
- ❌ Partial GGUF downloads  
- ❌ Layer sharding within GGUF files
- ❌ P2P/torrent distribution

**Each node still stores the full GGUF** - layer sharding will come in Task 8/9.

## 🚀 Success Criteria Met

✅ **Model Catalog:** Orchestrator shows supported models  
✅ **One-Command Install:** `POST /models/install` works  
✅ **Automatic Placement:** Layer distribution planned  
✅ **Smart Downloads:** Only where needed (infrastructure ready)  
✅ **Sharding Preparation:** `model_placement` structures in place

**Demo Workflow:**
```bash
# 1. Check available models
curl http://orchestrator:9000/catalog
# Returns: [{"id":"llama-3.2-1b", "size_gb":0.8, ...}]

# 2. Install model
curl -X POST http://orchestrator:9000/models/install \
  -d '{"model":"llama-3.2-1b"}'
# Returns: {"job_id":"install-123", "status":"started"}

# 3. Check installation
curl http://orchestrator:9000/models/install/install-123  
# Returns: {"status":"ready", "progress":1.0}

# 4. Use model (existing session API)
curl -X POST http://orchestrator:9000/session/create \
  -d '{"model":"llama-3.2-1b"}'
```

## 📁 Files Added/Modified

**New Files:**
- `tools/distributed/model_catalog.h` - Model management classes  
- `tools/distributed/model_catalog.cpp` - Implementation
- `tools/distributed/download_model.py` - HF downloader script
- `tools/distributed/test-model-catalog.cpp` - Unit tests
- `tools/distributed/test-model-install.cpp` - Integration tests
- `docs/task7_model_distribution.md` - This document

**Modified Files:**  
- `tools/distributed/orchestrator.cpp` - Added catalog APIs
- `tools/distributed/node_agent.cpp` - Added model store APIs
- `tools/distributed/CMakeLists.txt` - Build integration

## 🎯 Next Steps (Task 8/9)

Task 7 provides the foundation for:

1. **Task 8 - Distributed Prefill:** Nodes coordinate to split prefill computation
2. **Task 9 - Layer Sharding:** Nodes store only assigned layers, not full GGUF  
3. **Production Features:** Background downloads, progress UI, model validation

**Architecture is ready** for `git merge` workflows and scales to multiple models/nodes.