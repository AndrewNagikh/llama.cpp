#!/usr/bin/env bash
# Aggregate runtime traces from all Docker workers into one directory.
set -euo pipefail

TRACE_DIR="${LLAMA_DIST_TRACE_DIR:-/data/models/traces}"
OUT="${1:-./merged_traces}"

mkdir -p "$OUT"
for c in dist-node-a dist-node-b dist-node-c dist-orchestrator; do
  if docker ps --format '{{.Names}}' | grep -qx "$c"; then
    docker exec "$c" sh -c "test -d '$TRACE_DIR' && ls '$TRACE_DIR'/*.jsonl 2>/dev/null" | while read -r f; do
      base=$(basename "$f")
      docker cp "$c:$f" "$OUT/${c}_${base}"
    done || true
  fi
done

echo "Collected traces into $OUT"
python3 "$(dirname "$0")/merge_runtime_trace.py" "$OUT" -o "$OUT/timeline.json"
echo "Timeline: $OUT/timeline.json"
