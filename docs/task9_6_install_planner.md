# Task 9.6 — Install Planner & Download Operations

## Цель

Построить **детерминированный Install Plan** — описание действий для приведения кластера из Actual State в Desired State.

Planner получает Registry state (Manifest, Desired Layout, Actual Layout, Coverage) и возвращает список операций **без выполнения загрузок**.

---

## Модуль

```
tools/distributed/orchestrator/install_planner/
  install_planner.h
  install_planner.cpp
```

**Зависимости:** Model Registry, Manifest, Desired Layout, Coverage.  
**Не зависит от:** Hugging Face, HTTP, Downloader, Builder, Node Agent.

---

## Структуры

### `install_action`
`DOWNLOAD` | `VERIFY` | `DELETE` | `REPAIR`

### `download_operation`
- `layer_index`, `node_id`
- `tensor_offset`, `tensor_length` — из Manifest
- `source_url`, `checksum`

### `install_operation`
`action` + `node_id` + `layer_index` + опциональный `download`

### `install_plan`
- `operations[]` — плоский список
- `groups[]` — сгруппированные последовательные слои на ноде
- `operation_count`, `total_download_bytes`

---

## Алгоритм

Для каждого слоя Desired Layout:

| Coverage / Actual | Операция |
|-------------------|----------|
| READY на нужной ноде | ничего |
| MISSING | `DOWNLOAD` |
| CORRUPTED | `REPAIR` (с byte range) |
| READY на другой ноде | `DELETE` + `DOWNLOAD` |

**Byte range** вычисляется из `manifest.tensors` для слоя: `min(offset)` … `max(offset+size)`.

**Grouping:** последовательные слои с одинаковым `node_id` и `action` объединяются в `install_plan_group`.

---

## Registry

`dist_model_record.install_plan` — сохранённый план.  
`apply_install_plan()` — запись в Registry.

---

## REST API

### `POST /models/{id}/install-plan`
Опрашивает ноды, обновляет coverage, строит и сохраняет план.

### `GET /models/{id}/install-plan`
Возвращает сохранённый план.

Пример операции:

```json
{
  "action": "DOWNLOAD",
  "node": "node-a",
  "layer": 17,
  "offset": 8192331776,
  "length": 612123456
}
```

---

## Ограничения

- Нет HTTP / Range / скачивания GGUF
- Нет shard-файлов
- Node Agent не изменяется
- Install Plan независим от транспорта

---

## Тесты

### Unit
`test-install-plan`, `test-download-operation`, `test-repair-operation`, `test-delete-operation`, `test-operation-grouping`, `test-install-validation`

### Integration
`test-build-install-plan`, `test-install-plan-after-node-loss`, `test-install-plan-update`

### E2E
После Coverage: `build_install_plan` → `GET install-plan` → проверка `operation_count`, `offset/length`.

---

## E2E flow

```
Reconciliation / Coverage → Build Install Plan → GET Install Plan → Session Create
```

При `coverage READY`: `operation_count = 0`.  
После потери ноды: план содержит `DOWNLOAD` только для missing слоёв.
