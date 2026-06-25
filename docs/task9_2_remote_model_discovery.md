# Task 9.2 — Remote Model Discovery

## Goal

Add remote model discovery to the orchestrator. After this task the orchestrator can:

* connect to a remote **Model Provider**;
* query a model repository for available files, sizes, revisions and download URLs;
* store that metadata in the **Cluster Model Registry**;
* update the record when the remote revision changes.

**What is still forbidden:**

* downloading GGUF files;
* parsing GGUF headers / metadata / tensor directory;
* using HTTP Range requests.

Only the remote provider API is used.

---

## Architecture

A new module is added:

```text
tools/distributed/orchestrator/model_provider/
    model_provider.h
    model_provider.cpp
    huggingface_provider.h
    huggingface_provider.cpp
```

The orchestrator talks to providers only through the abstract interface:

```cpp
class model_provider {
public:
    virtual ~model_provider() = default;
    virtual std::string name() const = 0;
    virtual provider_discovery_result discover(const dist_model_record & model) = 0;
};
```

A small factory creates the concrete implementation from the `source` field of a registry record:

```cpp
std::unique_ptr<model_provider> create_model_provider(const std::string & name);
```

Currently only `"huggingface"` is supported.

---

## Data Structures

### `remote_model_file`

```cpp
struct remote_model_file {
    std::string filename;
    uint64_t    size_bytes = 0;
    std::string etag;
    std::string sha256;
    std::string download_url;
};
```

### `provider_discovery_result`

```cpp
struct provider_discovery_result {
    bool        success    = false;
    std::string provider;
    std::string repository;
    std::string revision;
    std::vector<remote_model_file> files;
    std::string error;
};
```

### Registry additions

`dist_model_record` is extended with discovery metadata:

```cpp
std::vector<remote_model_file> files;
std::string provider_revision;
std::string provider_etag;
std::chrono::system_clock::time_point last_discovery;
```

When a discovery succeeds the registry method `apply_discovery()`:

* stores the file list,
* stores the resolved revision,
* updates `last_discovery`,
* changes status from `DISCOVERED` to `MANIFEST_PENDING`.

The manifest itself is still intentionally empty — it will be created in Task 9.3.

---

## Hugging Face Provider

`huggingface_provider` uses the public Hugging Face Hub API:

```text
GET https://huggingface.co/api/models/{repository}/tree/{revision}
```

For every entry of `type == "file"` it records:

* `path` → `filename`
* `size` → `size_bytes`
* `oid` → `sha256` and `etag`
* `https://huggingface.co/{repository}/resolve/{revision}/{path}` → `download_url`

Behaviour:

* Missing or empty repository is rejected without a network call.
* `HF_TOKEN` is read from the environment and sent as `Authorization: Bearer ...` for private models.
* HTTP 401/403/404 and non-2xx responses are returned as graceful errors.
* An empty file list is reported as an error.
* No GGUF parsing, no Range requests, no file downloads.

---

## REST API

### `POST /models/{model_id}/discover`

Triggers discovery for an already registered model.

Request body: empty (the model is identified by the URL).

Success response:

```json
{
    "status": "ok",
    "provider": "huggingface",
    "files": 3,
    "revision": "main"
}
```

Possible errors:

* `404` — model not registered.
* `400` — unknown provider.
* `502` — provider returned an error (missing repo, network timeout, empty files, etc.).

### `GET /models/{model_id}`

Now includes discovery metadata:

```json
{
    "model_id": "llama-3.2-1b",
    "status": "MANIFEST_PENDING",
    "provider": "huggingface",
    "repository": "hugging-quants/Llama-3.2-1B-Instruct-Q4_K_M-GGUF",
    "revision": "main",
    "files": [
        {
            "filename": "llama-3.2-1b-instruct-q4_k_m.gguf",
            "size_bytes": 807690656,
            "etag": "...",
            "sha256": "...",
            "download_url": "https://huggingface.co/.../resolve/main/..."
        }
    ],
    "provider_revision": "main",
    "provider_etag": "main",
    "last_discovery": "2026-06-25T16:05:00Z"
}
```

### Existing endpoints

`POST /models/register`, `GET /models`, `GET /models/installed`, `DELETE /models/{model_id}` and the planner/session endpoints keep their Task 9.1 behaviour.

---

## Discovery Flow

```text
Register model  →  status = DISCOVERED
        ↓
POST /models/{id}/discover
        ↓
Orchestrator creates provider from record.source
        ↓
Provider queries remote API
        ↓
Registry.apply_discovery()
        ↓
status = MANIFEST_PENDING
files / revision / last_discovery stored
```

---

## Error Handling

The provider and endpoint were designed to never crash the orchestrator:

| Scenario                  | Result                                          |
|---------------------------|-------------------------------------------------|
| Unknown provider          | `400 unknown provider: ...`                     |
| Empty repository          | `502 repository is empty`                       |
| Repository not found      | `502 repository or revision not found`          |
| Unauthorized/private repo | `502 unauthorized: check HF_TOKEN ...`          |
| Network timeout           | `502 network error: could not reach ...`        |
| Empty file list           | `502 no files found in repository`              |
| Model not registered      | `404 model not registered`                      |

---

## Tests

### Unit tests

| Test                       | Purpose                                           |
|----------------------------|---------------------------------------------------|
| `test-provider-interface`  | Mock provider implements the abstract interface.  |
| `test-discovery-result`    | JSON round-trip for discovery result structures.  |
| `test-registry-discovery`  | Registry updates status and fields after discovery. |
| `test-huggingface-provider`| Real HF API call for a public model.              |

### Integration tests

| Test                       | Purpose                                           |
|----------------------------|---------------------------------------------------|
| `test-model-discovery`     | Register → discover → registry is MANIFEST_PENDING. |
| `test-discovery-errors`    | Unknown provider, missing repo, empty repo.       |
| `test-discovery-update`    | Re-discovery updates revision.                    |

### End-to-end tests

The discovery stage is inserted after model registration in:

* `test-orchestrator-dynamic-layout`
* `test-cluster-e2e`
* `test-cluster-e2e-node-loss`
* `test-cluster-e2e-install-reuse`
* `test-cluster-e2e-orchestrator-restart`

The E2E flow now is:

```text
Register model → Discover model → status == MANIFEST_PENDING → Session Create → ...
```

---

## Build & Run

From `llama.cpp/build`:

```bash
# build
 cmake --build . --target orchestrator \
       test-provider-interface test-discovery-result test-registry-discovery \
       test-huggingface-provider test-model-discovery test-discovery-errors \
       test-discovery-update test-model-registry test-model-status test-model-api \
       test-orchestrator-dynamic-layout test-cluster-e2e-node-loss \
       test-cluster-e2e-install-reuse test-cluster-e2e-orchestrator-restart -j4

# unit / integration tests
 ./bin/test-provider-interface
 ./bin/test-discovery-result
 ./bin/test-registry-discovery
 ./bin/test-huggingface-provider
 ./bin/test-model-discovery
 ./bin/test-discovery-errors
 ./bin/test-discovery-update
 ./bin/test-model-registry
 ./bin/test-model-status
 ./bin/test-model-api

# E2E
 ./bin/test-orchestrator-dynamic-layout ~/models/llama-3.2-1b-instruct-q4_k_m.gguf
 ./bin/test-cluster-e2e-node-loss
 ./bin/test-cluster-e2e-install-reuse
 ./bin/test-cluster-e2e-orchestrator-restart
```

---

## Limitations

* Only the Hugging Face provider is implemented.
* The manifest is still empty; it will be populated in Task 9.3.
* No GGUF download, parsing, or layer planning is performed.
