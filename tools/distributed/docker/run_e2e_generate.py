#!/usr/bin/env python3
"""Docker E2E: register → sync → generate across architecture families.

Uses all 3 Docker nodes (node-a/b/c). Retries sync until layer coverage is complete,
then session create + generate for models marked full_e2e=true.

Usage:
  ORCHESTRATOR=http://127.0.0.1:9000 python3 docker/run_e2e_generate.py
  ORCHESTRATOR=http://127.0.0.1:9000 python3 docker/run_e2e_generate.py --model tinyllama-1.1b
  ORCHESTRATOR=http://127.0.0.1:9000 python3 docker/run_e2e_generate.py --family gemma
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
ENV_FILE = ROOT / ".env"

ORCH = os.environ.get("ORCHESTRATOR", "http://127.0.0.1:9000")

MODELS = [
    {
        "label": "TinyLlama",
        "model_id": "tinyllama-1.1b",
        "family": "llama",
        "repository": "TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF",
        "filename": "tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf",
        "full_e2e": True,
    },
    {
        "label": "Llama3.2",
        "model_id": "llama-3.2-1b",
        "family": "llama",
        "repository": "hugging-quants/Llama-3.2-1B-Instruct-Q4_K_M-GGUF",
        "filename": "llama-3.2-1b-instruct-q4_k_m.gguf",
        "full_e2e": True,
    },
    {
        "label": "Qwen2.5",
        "model_id": "qwen2.5-1.5b",
        "family": "qwen",
        "repository": "Qwen/Qwen2.5-1.5B-Instruct-GGUF",
        "filename": "qwen2.5-1.5b-instruct-q4_k_m.gguf",
        "full_e2e": True,
    },
    {
        "label": "Gemma3",
        "model_id": "gemma-3-1b",
        "family": "gemma",
        "repository": "lmstudio-community/gemma-3-1b-it-GGUF",
        "filename": "gemma-3-1b-it-Q4_K_M.gguf",
        "full_e2e": True,
    },
    {
        "label": "Phi3.5",
        "model_id": "phi-3.5-mini",
        "family": "phi",
        "repository": "bartowski/Phi-3.5-mini-instruct-GGUF",
        "filename": "Phi-3.5-mini-instruct-Q4_K_M.gguf",
        "full_e2e": False,
    },
    {
        "label": "SmolLM2",
        "model_id": "smollm2-1.7b",
        "family": "smollm",
        "repository": "HuggingFaceTB/SmolLM2-1.7B-Instruct-GGUF",
        "filename": "smollm2-1.7b-instruct-q4_k_m.gguf",
        "full_e2e": False,
    },
    {
        "label": "DeepSeekDistill",
        "model_id": "deepseek-r1-distill-qwen-1.5b",
        "family": "deepseek",
        "repository": "unsloth/DeepSeek-R1-Distill-Qwen-1.5B-GGUF",
        "filename": "DeepSeek-R1-Distill-Qwen-1.5B-Q4_K_M.gguf",
        "full_e2e": False,
    },
]


def log(msg: str) -> None:
    print(msg, flush=True)


def load_hf_token() -> None:
    if os.environ.get("HF_TOKEN"):
        return
    if ENV_FILE.is_file():
        for line in ENV_FILE.read_text().splitlines():
            if line.strip().startswith("HF_TOKEN="):
                os.environ["HF_TOKEN"] = line.split("=", 1)[1].strip()
                return


def http(method: str, path: str, body: dict | None = None, timeout: int = 120) -> tuple[int, dict]:
    url = ORCH.rstrip("/") + path
    headers = {"Content-Type": "application/json", "Accept": "application/json"}
    token = os.environ.get("HF_TOKEN", "")
    if token:
        headers["Authorization"] = f"Bearer {token}"
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read().decode()
            return resp.status, json.loads(raw) if raw else {}
    except urllib.error.HTTPError as e:
        raw = e.read().decode()
        try:
            payload = json.loads(raw) if raw else {"error": str(e)}
        except json.JSONDecodeError:
            payload = {"error": raw or str(e)}
        return e.code, payload


def wait_job(job_id: str, timeout_s: int = 3600) -> tuple[bool, str]:
    log(f"    ... job {job_id} (timeout {timeout_s}s)")
    deadline = time.time() + timeout_s
    last_log = 0.0
    while time.time() < deadline:
        status, job = http("GET", f"/jobs/{job_id}", timeout=30)
        if status == 200:
            state = job.get("state", "")
            if time.time() - last_log > 30:
                nodes = job.get("nodes", {})
                failed = job.get("error", "")
                log(f"    ... job state={state} nodes={len(nodes)}" +
                    (f" err={failed[:60]}" if failed else ""))
                last_log = time.time()
            if state == "completed":
                return True, ""
            if state == "failed":
                return False, job.get("error", json.dumps(job))
        time.sleep(3)
    return False, "timeout"


def wait_cluster_nodes(min_count: int = 3, timeout_s: int = 60) -> bool:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        status, out = http("GET", "/nodes?format=brief", timeout=10)
        if status == 200:
            nodes = out.get("nodes", [])
            if len(nodes) >= min_count:
                return True
        time.sleep(2)
    return False


def model_record(model_id: str) -> dict:
    status, out = http("GET", f"/models/{model_id}", timeout=30)
    return out if status == 200 else {}


def refresh_coverage(model_id: str) -> dict:
    status, out = http("POST", f"/models/{model_id}/coverage/refresh", timeout=120)
    return out.get("coverage", {}) if status == 200 else {}


def layers_complete(cov: dict) -> bool:
    ready = cov.get("ready_layers", 0)
    total = cov.get("total_layers", 0)
    missing = cov.get("missing_layers", 0)
    state = cov.get("state", "")
    return (
        total > 0
        and ready == total
        and missing == 0
        and state == "READY"
    )


def install_plan_empty(model_id: str) -> bool:
    status, plan = http("POST", f"/models/{model_id}/install-plan", timeout=120)
    return status == 200 and plan.get("operation_count", 0) == 0


def sync_until_ready(model_id: str, max_rounds: int = 8) -> tuple[bool, str, dict]:
    last_cov: dict = {}
    for round_i in range(max_rounds):
        last_cov = refresh_coverage(model_id)
        state = last_cov.get("state", "")
        log(f"    coverage round {round_i + 1}: {state} "
            f"({last_cov.get('ready_layers', 0)}/{last_cov.get('total_layers', 0)} layers, "
            f"missing={last_cov.get('missing_layers', 0)})")

        if layers_complete(last_cov) and install_plan_empty(model_id):
            return True, state, last_cov

        status, plan = http("POST", f"/models/{model_id}/install-plan", timeout=120)
        if status != 200:
            return False, plan.get("error", json.dumps(plan)), last_cov
        ops = plan.get("operation_count", 0)
        if ops == 0:
            continue
        log(f"    install-plan ops={ops} bytes={plan.get('total_download_bytes', 0)}")

        status, out = http("POST", f"/models/{model_id}/install/execute", timeout=120)
        if status != 200:
            return False, out.get("error", json.dumps(out)), last_cov
        ok, err = wait_job(out.get("job_id", ""), timeout_s=3600)
        if not ok:
            log(f"    install job failed: {err[:120]}, retrying...")
            continue

    last_cov = refresh_coverage(model_id)
    ok = layers_complete(last_cov) and install_plan_empty(model_id)
    return ok, last_cov.get("state", ""), last_cov


def pipeline_model(cfg: dict) -> tuple[bool, str]:
    mid = cfg["model_id"]
    log(f"\n=== {cfg['label']} ({mid}) ===")

    http("POST", f"/models/{mid}/reset", {"keep_manifest": False}, timeout=180)

    status, out = http("POST", "/models/register", {
        "model_id": mid,
        "display_name": cfg["label"],
        "source": "huggingface",
        "repository": cfg["repository"],
        "filename": cfg["filename"],
        "revision": "main",
    })
    if status not in (200, 409) and model_record(mid).get("model_id") != mid:
        return False, f"register failed: {out}"

    for step, path in [
        ("discover", f"/models/{mid}/discover"),
        ("manifest", f"/models/{mid}/manifest"),
    ]:
        status, out = http("POST", path, {}, timeout=180)
        if status != 200:
            return False, f"{step}: {out}"

    status, out = http("POST", f"/models/{mid}/layout", {"force": True}, timeout=120)
    if status != 200:
        return False, f"layout: {out}"
    log(f"    layout placements={out.get('placements', '?')}")

    layout = model_record(mid).get("layout", {}).get("desired", {})
    placements = layout.get("placements", [])
    nodes = sorted({p.get("node_id", p.get("node", "")) for p in placements})
    log(f"    layout nodes: {nodes}")

    ok, detail, cov = sync_until_ready(mid)
    if not ok:
        missing = cov.get("missing", [])
        return False, f"sync/coverage: {detail} missing_layers={missing[:8]}"

    if not cfg.get("full_e2e", True):
        log(f"    sync-only PASS (family={cfg.get('family', '?')})")
        return True, "sync-only"

    status, out = http("POST", "/session/create", {"model": mid, "n_ctx": 512}, timeout=300)
    if status != 200:
        pipeline = out.get("layout", out.get("pipeline", ""))
        return False, f"session-create: {out.get('error', out)} pipeline={pipeline}"

    sid = out.get("session_id", "")
    pipeline = out.get("pipeline", [])
    log(f"    session {sid} pipeline={pipeline}")

    status, out = http("POST", "/session/generate", {
        "session_id": sid,
        "prompt": "The capital of France is",
        "max_tokens": 16,
    }, timeout=600)
    text = out.get("text", out.get("content", ""))
    if status != 200 or not text.strip():
        return False, f"generate: {out}"
    tokens = out.get("tokens", [])
    if tokens and len(set(tokens)) == 1 and len(tokens) >= 4:
        return False, f"repetitive tokens: {tokens[:8]}"
    log(f"    generate: {text!r}")
    return True, text


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", help="Run one model_id only")
    parser.add_argument("--family", help="Run all models in a family (llama, qwen, gemma, ...)")
    parser.add_argument("--full-e2e-only", action="store_true",
                        help="Only models with full distributed generate")
    args = parser.parse_args()

    load_hf_token()
    log(f"Orchestrator: {ORCH}")

    if not wait_cluster_nodes(3):
        log("Need 3 registered nodes")
        return 1

    models = MODELS
    if args.model:
        models = [m for m in MODELS if m["model_id"] == args.model]
        if not models:
            log(f"Unknown model: {args.model}")
            return 2
    elif args.family:
        models = [m for m in MODELS if m.get("family") == args.family]
        if not models:
            log(f"Unknown family: {args.family}")
            return 2
    if args.full_e2e_only:
        models = [m for m in models if m.get("full_e2e", True)]

    results: list[tuple[str, bool, str]] = []
    for cfg in models:
        try:
            ok, detail = pipeline_model(cfg)
        except Exception as exc:  # noqa: BLE001
            ok, detail = False, str(exc)
        results.append((cfg["label"], ok, detail))
        log(f"  => {'PASS' if ok else 'FAIL'}" + (f" — {detail[:150]}" if not ok else ""))

    log("\n" + "=" * 50)
    log("SUMMARY")
    for label, ok, detail in results:
        log(f"  {label:<12} {'PASS' if ok else 'FAIL'}" +
            (f" — {detail[:100]}" if not ok else ""))

    out_path = ROOT / "logs" / "docker_e2e_generate.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps([
        {"label": l, "ok": o, "detail": d} for l, o, d in results
    ], indent=2))
    log(f"\nSaved: {out_path}")

    return 0 if all(o for _, o, _ in results) else 1


if __name__ == "__main__":
    sys.exit(main())
