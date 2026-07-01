#!/usr/bin/env python3
"""Scale test: heavy model through Docker cluster with timing breakdown.

Usage:
  ORCHESTRATOR=http://127.0.0.1:9000 python3 run_scale_test.py
  ORCHESTRATOR=http://127.0.0.1:9000 python3 run_scale_test.py --model qwen3-8b
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

ROOT = Path(__file__).resolve().parents[3]  # llama.cpp
REPO_ROOT = ROOT.parent  # node-agent
ENV_FILE = REPO_ROOT / ".env"
ORCH = os.environ.get("ORCHESTRATOR", "http://127.0.0.1:9000")

SCALE_MODELS = {
    "qwen3-8b": {
        "label": "Qwen3-8B",
        "model_id": "qwen3-8b",
        "family": "qwen",
        "repository": "unsloth/Qwen3-8B-GGUF",
        "filename": "Qwen3-8B-Q4_K_M.gguf",
        "expected_size_gb": 5.03,
        "n_layer_hint": 36,
    },
}


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


def http(method: str, path: str, body: dict | None = None, timeout: int = 600) -> tuple[int, dict]:
    url = ORCH.rstrip("/") + path
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(url, data=data, method=method)
    if data is not None:
        req.add_header("Content-Type", "application/json")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read().decode()
            return resp.status, json.loads(raw) if raw else {}
    except urllib.error.HTTPError as e:
        raw = e.read().decode()
        try:
            return e.code, json.loads(raw)
        except json.JSONDecodeError:
            return e.code, {"error": raw}


def wait_cluster_nodes(min_nodes: int = 3, timeout_s: int = 120) -> bool:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        status, out = http("GET", "/nodes", timeout=15)
        if status == 200:
            online = [n for n in out.get("nodes", []) if n.get("online")]
            if len(online) >= min_nodes:
                log(f"Cluster: {len(online)} nodes online")
                for n in online:
                    mem = n.get("memory", {})
                    log(f"  {n['node_id']}: ram={mem.get('total_ram', 0)//1024//1024}MB "
                        f"free={mem.get('free_ram', 0)//1024//1024}MB score={n.get('score', 0):.1f}")
                return True
        time.sleep(2)
    return False


def job_progress_line(job: dict) -> str:
    nodes = job.get("nodes", {})
    if not nodes:
        return ""
    parts = []
    for nid, nd in sorted(nodes.items()):
        if isinstance(nd, dict):
            ready = nd.get("ready_count")
            total = nd.get("total_count")
            state = nd.get("state", nd.get("status", ""))
            if ready is not None and total is not None:
                parts.append(f"{nid}={ready}/{total}")
            elif state:
                parts.append(f"{nid}={state}")
    return " ".join(parts)


def wait_job(job_id: str, timeout_s: int | None = None) -> tuple[bool, str, float]:
    if timeout_s is None:
        timeout_s = int(os.environ.get("SYNC_JOB_TIMEOUT_S", "7200"))
    t0 = time.time()
    last_log = 0.0
    while time.time() - t0 < timeout_s:
        status, out = http("GET", f"/jobs/{job_id}", timeout=30)
        if status != 200:
            return False, str(out), time.time() - t0
        state = out.get("state", "")
        if time.time() - last_log > 15:
            progress = job_progress_line(out)
            log(f"  ... job state={state}" + (f" {progress}" if progress else ""))
            last_log = time.time()
        if state == "completed":
            return True, "", time.time() - t0
        if state == "failed":
            return False, out.get("error", state), time.time() - t0
        time.sleep(3)
    return False, "timeout", time.time() - t0


def coverage_from_refresh(body: dict) -> dict:
    cov = body.get("coverage", body)
    return cov if isinstance(cov, dict) else {}


def layers_complete(cov: dict) -> bool:
    ready = cov.get("ready_layers", 0)
    total = cov.get("total_layers", 0)
    missing = cov.get("missing_layers", 0)
    state = cov.get("state", "")
    return total > 0 and ready == total and missing == 0 and state == "READY"


def install_plan_empty(model_id: str) -> bool:
    status, plan = http("POST", f"/models/{model_id}/install-plan", timeout=120)
    return status == 200 and plan.get("operation_count", 0) == 0


def sync_until_ready(model_id: str, max_rounds: int = 20) -> tuple[bool, str, dict, float]:
    t0 = time.time()
    last_cov: dict = {}
    for round_i in range(max_rounds):
        status, out = http("POST", f"/models/{model_id}/coverage/refresh", timeout=120)
        if status != 200:
            return False, str(out), {}, time.time() - t0
        last_cov = coverage_from_refresh(out)
        state = last_cov.get("state", "")
        ready = last_cov.get("ready_layers", 0)
        total = last_cov.get("total_layers", 0)
        log(f"  coverage round {round_i + 1}: {state} ({ready}/{total}, missing={last_cov.get('missing_layers', 0)})")

        if layers_complete(last_cov) and install_plan_empty(model_id):
            return True, state, last_cov, time.time() - t0

        status, plan = http("POST", f"/models/{model_id}/install-plan", timeout=120)
        if status != 200:
            return False, str(plan), last_cov, time.time() - t0
        ops = plan.get("operation_count", 0)
        if ops == 0:
            continue
        log(f"  install-plan ops={ops} bytes={plan.get('total_download_bytes', 0)}")

        status, out = http("POST", f"/models/{model_id}/install/execute", timeout=120)
        if status != 200:
            return False, str(out), last_cov, time.time() - t0
        ok, err, job_s = wait_job(out.get("job_id", ""))
        log(f"  install job: {'OK' if ok else 'FAIL'} ({job_s:.0f}s) {err[:80] if err else ''}")
        if not ok:
            continue

    status, out = http("POST", f"/models/{model_id}/coverage/refresh", timeout=120)
    last_cov = coverage_from_refresh(out) if status == 200 else {}
    ok = layers_complete(last_cov) and install_plan_empty(model_id)
    return ok, last_cov.get("state", ""), last_cov, time.time() - t0


def run_scale_test(cfg: dict) -> dict:
    mid = cfg["model_id"]
    report: dict = {"model_id": mid, "label": cfg["label"], "timings": {}, "ok": False}

    log(f"\n{'='*60}\nScale test: {cfg['label']} ({mid})\n{'='*60}")

    t = time.time()
    http("POST", f"/models/{mid}/reset", {"keep_manifest": False}, timeout=180)
    report["timings"]["reset_s"] = round(time.time() - t, 1)

    t = time.time()
    http("POST", "/models/register", {
        "model_id": mid,
        "display_name": cfg["label"],
        "source": "huggingface",
        "repository": cfg["repository"],
        "filename": cfg["filename"],
        "revision": "main",
    })
    for step, path in [("discover", f"/models/{mid}/discover"), ("manifest", f"/models/{mid}/manifest")]:
        status, out = http("POST", path, {}, timeout=300)
        if status != 200:
            report["error"] = f"{step}: {out}"
            return report
    report["timings"]["discover_manifest_s"] = round(time.time() - t, 1)

    status, rec = http("GET", f"/models/{mid}", timeout=30)
    rec = rec or {}
    manifest = rec.get("manifest", {})
    report["architecture"] = manifest.get("architecture", "?")
    report["n_layer"] = manifest.get("n_layer", 0)
    report["n_embd"] = manifest.get("n_embd", 0)
    report["tensor_count"] = len(manifest.get("tensors", []))
    log(f"  manifest: arch={report['architecture']} layers={report['n_layer']} "
        f"embd={report['n_embd']} tensors={report['tensor_count']}")

    t = time.time()
    status, layout = http("POST", f"/models/{mid}/layout", {"force": True}, timeout=120)
    if status != 200:
        report["error"] = f"layout: {layout}"
        return report
    status, rec = http("GET", f"/models/{mid}", timeout=30)
    rec = rec or {}
    desired = (rec.get("layout") or {}).get("desired", {})
    placements = desired.get("placements", [])
    report["timings"]["layout_s"] = round(time.time() - t, 1)
    report["placements"] = len(placements)
    nodes_used = sorted({p.get("node", p.get("node_id", "")) for p in placements})
    report["layout_nodes"] = nodes_used
    mem_mb = desired.get("total_required_memory", layout.get("total_required_memory", 0)) // 1024 // 1024
    report["total_required_memory_mb"] = mem_mb
    log(f"  layout: {len(placements)} placements, nodes={nodes_used}, mem={mem_mb}MB")

    ok, detail, cov, sync_s = sync_until_ready(mid)
    report["timings"]["sync_s"] = round(sync_s, 1)
    report["coverage"] = cov
    if not ok:
        report["error"] = f"sync: {detail}"
        return report

    t = time.time()
    status, out = http("POST", "/session/create", {"model": mid, "n_ctx": 512}, timeout=600)
    report["timings"]["session_create_s"] = round(time.time() - t, 1)
    if status != 200:
        report["error"] = f"session: {out}"
        return report
    sid = out.get("session_id", "")
    pipeline = out.get("pipeline", [])
    report["session_id"] = sid
    report["pipeline"] = pipeline
    log(f"  session {sid} pipeline stages={len(pipeline)}")

    t = time.time()
    status, out = http("POST", "/session/generate", {
        "session_id": sid,
        "prompt": "The capital of France is",
        "max_tokens": 16,
    }, timeout=1200)
    gen_s = time.time() - t
    report["timings"]["generate_s"] = round(gen_s, 1)
    text = out.get("text", out.get("content", ""))
    tokens = out.get("tokens", [])
    report["generate_text"] = text
    report["generate_tokens"] = tokens
    if status != 200 or not text.strip():
        report["error"] = f"generate: {out}"
        return report
    if tokens and len(set(tokens)) == 1 and len(tokens) >= 4:
        report["error"] = f"repetitive tokens: {tokens[:8]}"
        return report

    tps = len(tokens) / gen_s if gen_s > 0 and tokens else 0
    report["decode_tps"] = round(tps, 2)
    report["ok"] = True
    log(f"  generate ({gen_s:.1f}s, {tps:.1f} tok/s): {text!r}")
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="qwen3-8b")
    parser.add_argument("--output", default="")
    args = parser.parse_args()

    load_hf_token()
    log(f"Orchestrator: {ORCH}")
    log(f"Host RAM: checking cluster node memory via /nodes")

    if not wait_cluster_nodes(3):
        log("ERROR: need 3 online nodes")
        return 1

    cfg = SCALE_MODELS.get(args.model)
    if not cfg:
        log(f"Unknown model: {args.model}")
        return 2

    report = run_scale_test(cfg)

    log(f"\n{'='*60}\nRESULT: {'PASS' if report.get('ok') else 'FAIL'}")
    if report.get("error"):
        log(f"  error: {report['error']}")
    log("Timings:")
    for k, v in report.get("timings", {}).items():
        log(f"  {k}: {v}s")
    if report.get("decode_tps"):
        log(f"  decode_tps: {report['decode_tps']}")

    out_path = Path(args.output) if args.output else REPO_ROOT / "logs" / f"scale_test_{args.model}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2, default=str))
    log(f"\nReport: {out_path}")

    return 0 if report.get("ok") else 1


if __name__ == "__main__":
    sys.exit(main())
