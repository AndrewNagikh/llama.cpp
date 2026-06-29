#!/usr/bin/env python3
"""Run mono + distributed debug parity for models on Docker cluster."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import urllib.error
import urllib.request
from pathlib import Path

PROMPT = "The capital of France is"
MAX_TOKENS = 16
ORCH = os.environ.get("ORCHESTRATOR", "http://127.0.0.1:9000")
TRACE_DIR = "/data/models/traces"

MODELS = [
    ("tinyllama-1.1b", "/data/models/tinyllama-1.1b/tokenizer.gguf"),
    ("llama-3.2-1b", "/data/models/llama-3.2-1b/tokenizer.gguf"),
    ("qwen2.5-1.5b", "/data/models/qwen2.5-1.5b/tokenizer.gguf"),
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


def http_post(path: str, body: dict, timeout: int = 300) -> tuple[int, dict]:
    url = ORCH.rstrip("/") + path
    req = urllib.request.Request(
        url,
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
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


def find_trace(container: str, session_id: str, role: str) -> str | None:
    code, out, _ = docker_exec(container, ["sh", "-c", f"ls {TRACE_DIR}/{session_id}_*_{role}.jsonl 2>/dev/null | head -1"])
    path = out.strip()
    return path if code == 0 and path else None


def docker_cp(container: str, remote: str, local: Path) -> None:
    local.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["docker", "cp", f"{container}:{remote}", str(local)], check=True)


def analyze(mono: Path, entry: Path, middle: Path | None, final: Path | None) -> dict:
    cmd = [
        "docker", "exec", "dist-node-a", "verify_decode_loop_parity",
        "--mono-trace", f"/tmp/{mono.name}",
        "--dist-entry", f"/tmp/{entry.name}",
    ]
    for c, p in [("dist-node-b", middle), ("dist-node-c", final)]:
        if p is not None:
            role = "middle" if c == "dist-node-b" else "final"
            cmd.extend([f"--dist-{role}", f"/tmp/{p.name}"])
            docker_cp(c, str(p), Path(f"/tmp/docker_debug/{p.name}"))
            subprocess.run(["docker", "cp", str(Path(f"/tmp/docker_debug/{p.name}")), f"dist-node-a:/tmp/{p.name}"], check=True)

    docker_cp("dist-node-a", str(mono), Path(f"/tmp/docker_debug/{mono.name}"))
    docker_cp("dist-node-a", str(entry), Path(f"/tmp/docker_debug/{entry.name}"))
    subprocess.run(["docker", "cp", str(Path(f"/tmp/docker_debug/{mono.name}")), f"dist-node-a:/tmp/{mono.name}"], check=True)
    subprocess.run(["docker", "cp", str(Path(f"/tmp/docker_debug/{entry.name}")), f"dist-node-a:/tmp/{entry.name}"], check=True)

    proc = subprocess.run(cmd, capture_output=True, text=True)
    try:
        return json.loads(proc.stdout)
    except json.JSONDecodeError:
        return {"status": "error", "stdout": proc.stdout, "stderr": proc.stderr, "code": proc.returncode}


def run_model(model_id: str, gguf_path: str, out_dir: Path) -> dict:
    log(f"\n{'='*60}\nDEBUG: {model_id}\n{'='*60}")
    session_tag = f"debug-{model_id.replace('.', '-')}"

    docker_exec("dist-node-a", ["mkdir", "-p", TRACE_DIR])

    code, out, err = docker_exec(
        "dist-node-a",
        ["mono_reference", gguf_path, PROMPT, str(MAX_TOKENS), session_tag],
        timeout=300,
    )
    log(f"  mono_reference: rc={code}")
    if code != 0:
        return {"model_id": model_id, "ok": False, "stage": "mono", "error": err or out}

    mono_path = f"{TRACE_DIR}/{session_tag}_local_monolithic.jsonl"
    code, out, _ = docker_exec("dist-node-a", ["test", "-f", mono_path])
    if code != 0:
        return {"model_id": model_id, "ok": False, "stage": "mono_trace", "error": "mono trace missing"}

    status, sess = http_post("/session/create", {"model": model_id, "n_ctx": 512})
    if status != 200:
        return {"model_id": model_id, "ok": False, "stage": "session", "error": sess}

    sid = sess.get("session_id", "")
    log(f"  session: {sid}")

    status, gen = http_post("/session/generate", {
        "session_id": sid,
        "prompt": PROMPT,
        "max_tokens": MAX_TOKENS,
    }, timeout=600)
    text = gen.get("text", "")
    tokens = gen.get("tokens", [])
    log(f"  generate: {text!r} tokens={tokens[:8]}...")
    if status != 200:
        return {"model_id": model_id, "ok": False, "stage": "generate", "error": gen}

    entry = find_trace("dist-node-a", sid, "entry")
    middle = find_trace("dist-node-b", sid, "middle")
    final = find_trace("dist-node-c", sid, "final")
    pipeline = find_trace("dist-node-a", sid, "pipeline")

    log(f"  traces: entry={entry} middle={middle} final={final} pipeline={pipeline}")

    if not entry or not final:
        code, listing, _ = docker_exec("dist-node-a", ["sh", "-c", f"ls -la {TRACE_DIR}/ 2>/dev/null | tail -20"])
        return {
            "model_id": model_id,
            "ok": False,
            "stage": "traces",
            "error": "missing worker traces",
            "listing": listing,
            "generate_text": text,
        }

    model_out = out_dir / model_id
    model_out.mkdir(parents=True, exist_ok=True)

    local_mono = model_out / "mono.jsonl"
    local_entry = model_out / "entry.jsonl"
    local_middle = model_out / "middle.jsonl" if middle else None
    local_final = model_out / "final.jsonl"

    docker_cp("dist-node-a", mono_path, local_mono)
    docker_cp("dist-node-a", entry, local_entry)
    docker_cp("dist-node-c", final, local_final)
    if middle:
        docker_cp("dist-node-b", middle, local_middle)
        subprocess.run(["docker", "cp", str(local_middle), f"dist-node-a:/tmp/{local_middle.name}"], check=True)
    subprocess.run(["docker", "cp", str(local_mono), f"dist-node-a:/tmp/{local_mono.name}"], check=True)
    subprocess.run(["docker", "cp", str(local_entry), f"dist-node-a:/tmp/{local_entry.name}"], check=True)
    subprocess.run(["docker", "cp", str(local_final), f"dist-node-a:/tmp/{local_final.name}"], check=True)

    cmd = [
        "docker", "exec", "dist-node-a", "verify_decode_loop_parity",
        "--mono-trace", f"/tmp/{local_mono.name}",
        "--dist-entry", f"/tmp/{local_entry.name}",
    ]
    if middle:
        cmd.extend(["--dist-middle", f"/tmp/{local_middle.name}"])
    cmd.extend(["--dist-final", f"/tmp/{local_final.name}"])

    proc = subprocess.run(cmd, capture_output=True, text=True)
    try:
        report = json.loads(proc.stdout)
    except json.JSONDecodeError:
        report = {"status": "parse_error", "stdout": proc.stdout, "stderr": proc.stderr}

    log(f"  parity: {report.get('status')} — {report.get('root_cause', report.get('message', ''))}")

    return {
        "model_id": model_id,
        "ok": report.get("status") == "ok",
        "session_id": sid,
        "generate_text": text,
        "tokens": tokens,
        "report": report,
        "trace_dir": str(model_out),
    }


def main() -> int:
    out_dir = Path(os.environ.get("DEBUG_OUT", "/Users/user/Documents/node-agent/logs/docker_debug"))
    out_dir.mkdir(parents=True, exist_ok=True)

    results = []
    for model_id, gguf in MODELS:
        try:
            results.append(run_model(model_id, gguf, out_dir))
        except Exception as exc:
            results.append({"model_id": model_id, "ok": False, "error": str(exc)})

    summary_path = out_dir / "debug_parity_summary.json"
    summary_path.write_text(json.dumps(results, indent=2))

    log("\n" + "=" * 60)
    log("SUMMARY")
    for r in results:
        status = "PASS" if r.get("ok") else "DIVERGENCE" if r.get("report") else "FAIL"
        rc = r.get("report", {}).get("root_cause", r.get("error", ""))
        log(f"  {r['model_id']:<18} {status}  {str(rc)[:80]}")
    log(f"\nSaved: {summary_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
