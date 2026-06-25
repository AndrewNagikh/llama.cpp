# Task 9.5 — Cluster Coverage & Reconciliation

## Цель

Отслеживать **фактическое состояние** модели в кластере и сравнивать его с **целевым** (Desired Layout).

Orchestrator после Task 9.5 знает:

- какие слои должны быть установлены (Desired);
- какие слои реально установлены (Actual);
- какие отсутствуют или повреждены;
- что требует установки или восстановления (через Reconciliation).

На этом этапе **не скачиваются модели**, не создаются shard-файлы, не выполняется автоматическое восстановление.

---

## Архитектура

### Модуль

```
tools/distributed/orchestrator/coverage/
  coverage.h
  coverage.cpp
```

Зависимости:

- Cluster Model Registry
- Desired Layout (Task 9.4)
- Node reports (`GET /installed-layers`)

Не зависит от Downloader или Manifest Builder.

---

## Два независимых состояния

### Desired State

Источник: Layout Planner (Task 9.4).  
Хранится в `dist_model_record.layout`.

Planner **никогда** не изменяет Actual State.

### Actual State

Источник: Node Agent (`GET /installed-layers`).  
Хранится в `dist_model_record.actual`.

Node Agent **никогда** не изменяет Desired State.

---

## Основные структуры

### `install_state`

`MISSING` | `DOWNLOADING` | `VERIFYING` | `READY` | `CORRUPTED`

### `installed_layer`

Пер-слойное фактическое состояние на ноде: индекс, node_id, device, size_bytes, checksum, state, updated_at.

### `actual_model_layout`

Список `installed_layer` для модели (агрегация отчётов всех нод).

### `coverage_state`

| Состояние | Условие |
|-----------|---------|
| `EMPTY` | Нет готовых слоёв |
| `PARTIAL` | Часть слоёв READY, есть MISSING |
| `READY` | Все desired-слои READY на нужных нодах |
| `DEGRADED` | Есть CORRUPTED или MISSING на недоступной ноде |

### `coverage_report`

Сводка: total/ready/missing/corrupted + списки индексов.

### `reconciliation_result`

Списки `missing` и `corrupted` для следующих Task (Install Plan).

---

## Алгоритм Coverage

Для каждого `layer_placement` из Desired Layout:

1. Найти `installed_layer` с тем же `layer_index` и `node_id`.
2. Если нода не ответила на poll → `MISSING` (DEGRADED при потере ноды).
3. Если слой не найден → `MISSING`.
4. Если `state == CORRUPTED` → `CORRUPTED`.
5. Если `state == READY` и device совпадает → `READY`.

Валидация:

- каждый desired-слой учтён;
- нет лишних actual-слоёв (`coverage_has_no_extra_layers`);
- checksum/state учитываются при определении CORRUPTED.

---

## Reconciliation

`reconcile_layers(desired, actual, online_nodes)` возвращает:

```json
{
  "missing": [17, 18, 19],
  "corrupted": [33],
  "state": "DEGRADED"
}
```

Скачивание и установка **не выполняются** — только список различий.

---

## Node Agent API

### `GET /installed-layers?model={id}`

```json
{
  "model": "llama-3.2-1b",
  "layers": [
    { "layer": 0, "node": "node-a", "device": "metal", "checksum": "stub:...", "state": "READY" }
  ]
}
```

Пока данные — **заглушки**: при установленной модели (`model_store.ready`) нода сообщает все слои как `READY` с stub-checksum.

---

## Orchestrator REST API

### `POST /models/{id}/coverage/refresh`

Опрашивает ноды, обновляет Actual + Coverage в Registry.

### `GET /models/{id}/coverage`

Возвращает сохранённый `coverage_report`.

### `POST /models/{id}/reconcile`

Опрашивает ноды, пересчитывает coverage, возвращает `reconciliation_result`.

---

## Registry

`dist_model_record` хранит:

- `layout` — Desired
- `actual` — Actual
- `coverage` — последний Coverage Report

Методы:

- `apply_actual()`
- `apply_coverage()`
- `refresh_coverage()`

---

## Тесты

### Unit

`test-coverage`, `test-reconcile`, `test-checksum`, `test-empty-cluster`, `test-partial-coverage`, `test-ready-coverage`, `test-degraded-coverage`

### Integration

`test-node-report`, `test-coverage-refresh`, `test-reconcile-after-node-loss`

### E2E

```
Build Layout → Install → Coverage READY → Session Create
```

После потери ноды:

```
Reconcile → DEGRADED + missing layers
```

---

## Ограничения

Запрещено на этом этапе:

- HTTP Range / скачивание GGUF;
- создание shard;
- автоматическая установка отсутствующих слоёв;
- изменение runtime `POST /session/create`.

Task заканчивается вычислением списка различий.
