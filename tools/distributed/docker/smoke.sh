#!/usr/bin/env bash
# Quick health check for Docker cluster (orchestrator + 3 nodes).
set -euo pipefail

ORCH="${ORCH_URL:-http://127.0.0.1:9000}"
MAX_WAIT="${SMOKE_WAIT_SEC:-90}"

echo "==> orchestrator health: $ORCH"
deadline=$((SECONDS + MAX_WAIT))
until curl -sf "$ORCH/health" >/dev/null; do
  if (( SECONDS >= deadline )); then
    echo "FAIL: orchestrator not healthy after ${MAX_WAIT}s"
    exit 1
  fi
  sleep 2
done
echo "OK orchestrator /health"

echo "==> waiting for 3 registered nodes..."
while (( SECONDS < deadline )); do
  nodes_json=$(curl -sf "$ORCH/nodes?format=brief" || echo '{"nodes":[]}')
  count=$(python3 -c "import json,sys; d=json.loads(sys.argv[1]); print(len(d.get('nodes', d if isinstance(d,list) else [])))" "$nodes_json" 2>/dev/null || echo 0)
  if [[ "$count" -ge 3 ]]; then
    echo "OK nodes registered: $count"
    echo "$nodes_json" | python3 -m json.tool
    exit 0
  fi
  sleep 2
done

echo "FAIL: expected 3 nodes, got:"
curl -sf "$ORCH/nodes?format=brief" | python3 -m json.tool || true
exit 1
