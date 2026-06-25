# Task 9.1 — Cluster Model Registry

## Цель

Добавить в orchestrator полноценный **Cluster Model Registry**.

Registry становится единственным источником информации о моделях в кластере:
- `POST /session/create` больше не принимает произвольное имя модели;
- сначала модель должна быть зарегистрирована через `POST /models/register`;
- если модель не зарегистрирована — возвращается `HTTP 404 model not registered`.

На этом этапе **не происходит скачивания моделей**, не используется Hugging Face API, GGUF parser и HTTP Range. Это чисто инфраструктурный этап.

---

## Архитектура

### Модуль

```
llama.cpp/tools/distributed/orchestrator/model_registry.h
llama.cpp/tools/distributed/orchestrator/model_registry.cpp
```

Модуль полностью изолирован: он не зависит от `llama.h`, GGUF, сетевых загрузок и Node Agent.

### Структура записи

```cpp
struct dist_model_record {
    std::string model_id;
    std::string display_name;
    std::string source;
    std::string repository;
    std::string filename;
    std::string revision;
    std::string architecture;
    dist_model_status status = dist_model_status::discovered;
    std::optional<model_manifest> manifest;
};
```

Поле `manifest` оставлено пустым — оно будет заполняться позже.

### Статусы модели

```cpp
DISCOVERED
MANIFEST_PENDING
MANIFEST_READY
INSTALLING
PARTIALLY_AVAILABLE
AVAILABLE
DEGRADED
UNAVAILABLE
```

На этапе 9.1 реально используется только `DISCOVERED`.

---

## REST API

### `POST /models/register`

Регистрирует модель или обновляет существующую запись.

Пример тела:

```json
{
    "model_id": "llama-3.2-1b",
    "display_name": "Llama 3.2 1B Q4_K_M",
    "source": "huggingface",
    "repository": "hugging-quants/Llama-3.2-1B-Instruct-Q4_K_M-GGUF",
    "filename": "llama-3.2-1b-instruct-q4_k_m.gguf",
    "revision": "main"
}
```

Статус принудительно устанавливается в `DISCOVERED`.

### `GET /models`

Возвращает список всех зарегистрированных моделей.

### `GET /models/{model_id}`

Возвращает полную информацию об одной модели.

### `DELETE /models/{model_id}`

Удаляет запись из Registry. Никаких действий на нодах не выполняется.

### `GET /models/installed`

Старый эндпоинт `GET /models`, агрегирующий установленные модели с нод, переехал сюда.

---

## Интеграция с планировщиком

`POST /session/create`, `POST /planner/simulate` и `GET /planner/explain` теперь:

1. Ищут `model_id` в `cluster_model_registry`.
2. Если модель не найдена — возвращают `HTTP 404 {"error":"model not registered"}`.
3. Если найдена — используют запись для оценки памяти:
   - сначала пытаются найти локальный GGUF по полю `filename` (`--model` / `--models-dir`);
   - если GGUF не найден, делают fallback на старый catalog до появления `manifest`.

---

## Node Agent

На этом этапе Node Agent не изменяется. Он ничего не знает про Registry.

---

## CLI orchestrator

Добавлена опция `--models-dir DIR`:

```bash
./orchestrator --model ~/models/llama-3.2-1b-instruct-q4_k_m.gguf \
               --models-dir ~/models \
               --listen 0.0.0.0:9000
```

Если `--models-dir` не указан, по умолчанию используется директория файла `--model`.

---

## Тесты

### Unit

```bash
cd llama.cpp/build
./bin/test-model-registry
./bin/test-model-status
./bin/test-model-api
```

### Интеграционные / E2E

```bash
# test-cluster-e2e теперь перед установкой модели регистрирует её в Registry
./bin/test-cluster-e2e --gguf ~/models/llama-3.2-1b-instruct-q4_k_m.gguf

./bin/test-orchestrator-dynamic-layout ~/models/llama-3.2-1b-instruct-q4_k_m.gguf
./bin/test-cluster-e2e-node-loss
./bin/test-cluster-e2e-orchestrator-restart
./bin/test-cluster-e2e-install-reuse
```

---

## Критерий готовности

- [x] Есть полноценный `cluster_model_registry`.
- [x] `POST /session/create` использует Registry и возвращает `404`, если модель не зарегистрирована.
- [x] Работают `POST /models/register`, `GET /models`, `GET /models/{id}`, `DELETE /models/{id}`.
- [x] Unit tests проходят.
- [x] Обновлён общий E2E — добавлен шаг регистрации модели.
- [x] Существующий distributed inference не сломан.
