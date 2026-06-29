#!/usr/bin/env python3
"""Run runtime state + hidden transport verification in Docker."""

from __future__ import annotations

import json
import os
import subprocess
import sys

PROMPT = os.environ.get("STATE_VERIFY_PROMPT", "The capital of France is")
MODELS = [
    ("qwen2.5-1.5b", "/data/models/qwen2.5-1.5b/tokenizer.gguf"),
    ("tinyllama-1.1b", "/data/models/tinyllama-1.1b/tokenizer.gguf"),
]


def docker_exec(cmd: list[str], timeout: int = 600) -> tuple[int, str, str]:
    proc = subprocess.run(
        ["docker", "exec", "dist-node-a"] + cmd,
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    return proc.returncode, proc.stdout, proc.stderr


def main() -> int:
    results = []

    # TCP loopback
    code, out, err = docker_exec(["verify_hidden_transport", "4", "256"])
    results.append({"check": "tcp_loopback", "code": code, "out": out.strip()})

    for model_id, gguf in MODELS:
        code, out, _ = docker_exec(["test", "-f", gguf])
        if code != 0:
            continue
        code, out, err = docker_exec(["verify_runtime_api", gguf, "1"])
        results.append({"check": f"runtime_api_{model_id}", "code": code, "out": out.strip()})

    print(json.dumps(results, indent=2))
    failed = [r for r in results if r["code"] != 0]
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
