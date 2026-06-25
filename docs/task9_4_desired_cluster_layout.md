# Task 9.4 — Desired Cluster Layout Planner

## Цель

Реализовать планировщик **целевого размещения модели** (Desired State) в кластере.

Planner определяет:

- какие слои на какой ноде;
- на каком устройстве (CPU / CUDA / Metal);
- хватает ли памяти;
- какой Install Plan потребуется на следующих этапах.

На этом этапе **не происходит** скачивание модели, HTTP Range, создание shard-файлов и изменение runtime `POST /session/create`.

---

## Модуль

```
tools/distributed/orchestrator/layout_planner/
  layout_planner.h
  layout_planner.cpp
```

Зависимости:

- Cluster Model Registry (Task 9.1)
- Model Manifest (Task 9.3)
- Memory Planner / `memory_estimator` (Task 8)
- Node capabilities и benchmark (через `dist_node_info`)

Planner **не** зависит от downloader или installer.

---

## Основные сущности

### `layer_placement`

Размещение одного слоя:

| Поле | Описание |
|------|----------|
| `layer_index` | Индекс слоя (0 … n_layer-1) |
| `node_id` | ID ноды |
| `device` | `cpu`, `cuda` или `metal` |
| `size_bytes` | Размер весов слоя из Manifest |
| `required` | Обязательность слоя (по умолчанию `true`) |

### `desired_model_layout`

Целевое состояние размещения модели:

- `placements` — по одной записи на каждый слой;
- `total_weight_bytes` — сумма весов;
- `total_required_memory` — веса + KV + compute + scratch;
- `fits_cluster` — помещается ли модель в кластер;
- `warnings` — предупреждения (например, нехватка памяти).

### `model_layout`

Обёртка для Registry:

```cpp
struct model_layout {
    desired_model_layout desired;
};
```

Registry владеет `std::optional<model_layout> layout` в `dist_model_record`.

---

## REST API

### `POST /models/{id}/layout`

Строит Desired Layout из Manifest и зарегистрированных online-нод.

Тело (опционально):

```json
{ "n_ctx": 4096 }
```

Ответ при успехе:

```json
{
  "status": "ok",
  "model": "qwen3-70b",
  "fits_cluster": true,
  "placements": 80,
  "total_weight_bytes": 42000000000,
  "total_required_memory": 45000000000
}
```

### `GET /models/{id}/layout`

Возвращает сохранённый layout:

```json
{
  "model": "qwen3-70b",
  "fits_cluster": true,
  "placements": [
    { "layer": 0, "node": "node-a", "device": "cuda", "size_mb": 612 },
  ]
}
```

---

## Алгоритм

1. **Проверка входа** — Manifest с layer descriptors, хотя бы одна нода с ненулевым бюджетом памяти.

2. **Расчёт памяти** — `memory_requirements_from_manifest()` использует реальные `size_bytes` каждого слоя из Manifest (не равномерное распределение).

3. **Проверка кластера** — суммарный бюджет RAM/VRAM всех нод сравнивается с `total_required_memory`.

4. **Сортировка нод** — GPU-ноды первыми, затем по benchmark `score` (быстрее = выше приоритет).

5. **Распределение слоёв** — score-proportional allocation:
   - доля слоёв пропорциональна score;
   - минимум один слой на активную ноду (если слоёв ≥ числа нод);
   - clamp по per-node memory cap (сумма `layer_placement_cost` не превышает бюджет).

6. **Выбор устройства** — GPU-ноды → `cuda` или `metal` по backend; CPU-only → `cpu`.

7. **Построение placements** — для каждого слоя: node, device, `size_bytes` из Manifest.

8. **Валидация**:
   - каждый слой ровно один раз;
   - нет пропусков и дубликатов;
   - сумма `size_bytes` совпадает с Manifest.

9. **Сохранение** — `cluster_model_registry::apply_layout()`.

---

## Входные данные Planner

```
Model Manifest
    ↓
Node Capabilities (backend, has_gpu)
    ↓
Node Benchmark (score)
    ↓
Memory (free RAM / VRAM)
    ↓
Context Size (n_ctx)
```

---

## Тесты

### Unit

| Тест | Проверка |
|------|----------|
| `test-layout-planner` | Построение layout на 3 нодах |
| `test-layer-placement` | Корректность каждого placement |
| `test-memory-fit` | Отказ при нехватке памяти |
| `test-no-overlap` | Нет дубликатов слоёв |
| `test-full-coverage` | Все слои присутствуют |
| `test-gpu-priority` | Быстрый GPU получает больше слоёв |

### Integration

| Тест | Проверка |
|------|----------|
| `test-build-layout` | Manifest → Layout → Registry |
| `test-layout-update` | Пересчёт после уменьшения памяти |
| `test-layout-node-loss` | `fits_cluster=false` после потери ноды |

### E2E

После `build_manifest` добавлен этап `build_layout`:

```
Build Manifest → Build Desired Layout → GET Layout → Install → Session Create
```

Проверяется: полное покрытие слоёв, отсутствие overlap, `fits_cluster == true`.

---

## Ограничения

Запрещено на этом этапе:

- скачивание модели;
- HTTP Range;
- создание shard;
- изменение Node Agent;
- установка модели;
- изменение runtime planner в `POST /session/create`.

Planner только строит Desired State для последующего Install Plan.
