#!/usr/bin/env python3
"""Run verify_layer_equivalence (mono vs worker_entry) inside Docker cluster."""

from __future__ import annotations

import json
import os
import subprocess
import sys

PROMPT = os.environ.get("LAYER_EQ_PROMPT", "The capital of France is")
TRACE_DIR = os.environ.get("LLAMA_LAYER_TRACE_DIR", "/data/models/layer_trace")

MODELS = [
    ("tinyllama-1.1b", "/data/models/tinyllama-1.1b"),
    ("llama-3.2-1b", "/data/models/llama-3.2-1b"),
    ("qwen2.5-1.5b", "/data/models/qwen2.5-1.5b"),
]


def log(msg: str) -> None:
    print(msg, flush=True)


def docker_exec(container: str, cmd: list[str], timeout: int = 600) -> tuple[int, str, str]:
    proc = subprocess.run(
        ["docker", "exec", container] + cmd,
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    return proc.returncode, proc.stdout, proc.stderr


def run_model(model_id: str, base: str) -> dict:
    mono = f"{base}/tokenizer.gguf"
    worker = f"{base}/worker_entry.gguf"
    log(f"\n{'='*60}\nLAYER EQ: {model_id}\n{'='*60}")

    for path in (mono, worker):
        code, _, _ = docker_exec("dist-node-a", ["test", "-f", path])
        if code != 0:
            return {"model_id": model_id, "ok": False, "error": f"missing {path}"}

    docker_exec("dist-node-a", ["mkdir", "-p", TRACE_DIR])
    env = [
        "env",
        "LLAMA_LAYER_TRACE=1",
        f"LLAMA_LAYER_TRACE_DIR={TRACE_DIR}/{model_id}",
        "LLAMA_GRAPH_DUMP=1",
    ]
    cmd = env + [
        "verify_layer_equivalence",
        mono,
        worker,
        PROMPT,
    ]
    code, out, err = docker_exec("dist-node-a", cmd, timeout=900)
    log(out)
    if err.strip():
        log(err)

    try:
        report = json.loads(out)
    except json.JSONDecodeError:
        return {
            "model_id": model_id,
            "ok": False,
            "code": code,
            "stdout": out,
            "stderr": err,
        }

    report["model_id"] = model_id
    report["ok"] = report.get("all_pass", False)
    report["exit_code"] = code
    return report


def main() -> int:
    if len(sys.argv) > 1:
        selected = {sys.argv[1]}
    else:
        selected = {m[0] for m in MODELS}

    results = []
    for model_id, base in MODELS:
        if model_id not in selected:
            continue
        results.append(run_model(model_id, base))

    summary_path = "/tmp/layer_equivalence_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    log(f"\nWrote {summary_path}")
    log(json.dumps(results, indent=2))

    failed = [r for r in results if not r.get("ok")]
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
