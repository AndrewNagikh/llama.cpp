#!/usr/bin/env python3
"""Task 9.8.6 regression — final runtime isolation across models."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

PROMPT = os.environ.get("LLAMA_TEST_PROMPT", "The capital of France is")
OUT_DIR = Path(os.environ.get("LLAMA_FINAL_RUNTIME_OUT", "/data/logs/final_runtime"))
MODELS_DIR = Path(os.environ.get("MODELS_DIR", "/data/models"))

MODELS = [
    {
        "name": "TinyLlama",
        "mono": "TinyLlama-1.1B-Chat-v1.0/tokenizer.gguf",
        "worker_final": "TinyLlama-1.1B-Chat-v1.0/worker_final.gguf",
        "layer_boundary": 16,
    },
    {
        "name": "Llama-3.2",
        "mono": "Llama-3.2-1B-Instruct/tokenizer.gguf",
        "worker_final": "Llama-3.2-1B-Instruct/worker_final.gguf",
        "layer_boundary": 10,
    },
    {
        "name": "Qwen2.5",
        "mono": "Qwen2.5-0.5B-Instruct/tokenizer.gguf",
        "worker_final": "Qwen2.5-0.5B-Instruct/worker_final.gguf",
        "layer_boundary": 19,
    },
]


def run_verify(mono: Path, worker: Path, layer_boundary: int, out: Path) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    cmd = [
        "verify_final_runtime",
        str(mono),
        str(worker),
        PROMPT,
        str(layer_boundary),
        "--out",
        str(out),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    report_path = out / "runtime_equivalence.json"
    if report_path.exists():
        return json.loads(report_path.read_text())
    if proc.stdout.strip():
        return json.loads(proc.stdout.strip())
    return {"all_pass": False, "message": proc.stderr or "verify_final_runtime failed"}


def main() -> int:
    summary = []
    failed = 0

    for spec in MODELS:
        mono = MODELS_DIR / spec["mono"]
        worker = MODELS_DIR / spec["worker_final"]
        out = OUT_DIR / spec["name"].lower().replace(".", "_").replace("-", "_")

        if not mono.exists() or not worker.exists():
            print(f"SKIP {spec['name']}: missing model files")
            continue

        report = run_verify(mono, worker, spec["layer_boundary"], out)
        ok = report.get("all_pass", False)
        if not ok and report.get("prefill_logits_pass") is False:
            failed += 1
        summary.append(
            {
                "model": spec["name"],
                "all_pass": ok,
                "prefill_logits_pass": report.get("prefill_logits_pass"),
                "root_cause_field": report.get("root_cause_field"),
                "message": report.get("message"),
                "report": str(out / "runtime_equivalence.json"),
            }
        )
        print(json.dumps(summary[-1], indent=2))

    summary_path = OUT_DIR / "final_runtime_summary.json"
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"Wrote {summary_path}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
