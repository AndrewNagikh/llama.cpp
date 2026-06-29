#!/usr/bin/env python3
"""Run monolithic reference + distributed generate with debug traces in Docker."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import urllib.request
from pathlib import Path

PROMPT = "The capital of France is"
MODELS = {
    "tinyllama-1.1b": "tinyllama-1.1b",
    "llama-3.2-1b": "llama-3.2-1b",
    "qwen2.5-1.5b": "qwen2.5-1.5b",
}


def http_post(url: str, path: str, body: dict, timeout: int = 300) -> dict:
    req = urllib.request.Request(
        url.rstrip("/") + path,
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode())


def run_model(model_id: str, orch: str, trace_dir: Path) -> dict:
    print(f"\n=== debug parity: {model_id} ===")
    sess = http_post(orch, "/session/create", {"model": model_id, "n_ctx": 512})
    sid = sess.get("session_id", "")
    gen = http_post(orch, "/session/generate", {
        "session_id": sid,
        "prompt": PROMPT,
        "max_tokens": 16,
    }, timeout=600)
    print(f"  generate: {gen.get('text', '')!r}")

    traces = sorted(trace_dir.glob(f"{sid}_*.jsonl"))
    mono_candidates = sorted(trace_dir.glob("mono_ref_*.jsonl"))
    return {
        "model_id": model_id,
        "session_id": sid,
        "generate": gen,
        "dist_traces": [str(p) for p in traces],
        "mono_traces": [str(p) for p in mono_candidates],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--orchestrator", default=os.environ.get("ORCHESTRATOR", "http://127.0.0.1:9000"))
    parser.add_argument("--models", nargs="*", default=list(MODELS.keys()))
    parser.add_argument("--trace-dir", default=os.environ.get("LLAMA_DIST_TRACE_DIR", ""))
    args = parser.parse_args()

    trace_dir = Path(args.trace_dir) if args.trace_dir else None
    results = []
    for mid in args.models:
        if mid not in MODELS:
            print(f"skip unknown model {mid}")
            continue
        if trace_dir is None:
            print("Set LLAMA_DIST_TRACE_DIR or --trace-dir")
            return 2
        results.append(run_model(mid, args.orchestrator, trace_dir))

    out = Path("logs/debug_parity_results.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2))
    print(f"\nSaved: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
