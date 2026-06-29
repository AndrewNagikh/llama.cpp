#!/usr/bin/env python3
"""Analyze runtime state + transport traces from Task 9.8.5 debug run."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


def load_jsonl(path: str) -> list[dict]:
    events = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                events.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return events


def memcmp_bins(a: str, b: str) -> tuple[bool, int]:
    with open(a, "rb") as fa, open(b, "rb") as fb:
        ba, bb = fa.read(), fb.read()
    if len(ba) != len(bb):
        return False, 0
    for i, (x, y) in enumerate(zip(ba, bb)):
        if x != y:
            return False, i
    return True, -1


def analyze_transport(trace_dir: str) -> list[dict]:
    rows = []
    before = sorted(Path(trace_dir).glob("transport_s*_ab_hidden_before_tcp.bin"))
    for bf in before:
        name = bf.name
        step = name.split("_")[1][1:]  # s0 -> 0
        link = "ab" if "_ab_" in name else "bc"
        af = bf.parent / name.replace("before_tcp", "after_tcp")
        if not af.exists():
            rows.append({"step": int(step), "link": link, "ok": False, "message": "missing after_tcp"})
            continue
        ok, off = memcmp_bins(str(bf), str(af))
        rows.append({
            "step": int(step),
            "link": link,
            "ok": ok,
            "bytes": bf.stat().st_size,
            "diff_byte": off if not ok else -1,
        })
    return rows


def summarize_trace(events: list[dict], role: str) -> list[dict]:
    rows = []
    for ev in events:
        et = ev.get("event")
        if et not in ("hidden", "runtime_state", "transport", "token_selected", "logits", "kv", "position"):
            continue
        row = {
            "event": et,
            "step": ev.get("step"),
            "phase": ev.get("phase"),
            "worker": ev.get("worker", role),
        }
        if et == "hidden":
            stats = ev.get("stats") or {}
            row["sha256"] = stats.get("sha256", "")[:16]
            row["mean"] = stats.get("mean")
        if et == "transport":
            tr = ev.get("transport") or {}
            row["link"] = tr.get("link")
            row["memcmp_ok"] = tr.get("memcmp_ok")
            row["latency_ms"] = tr.get("latency_ms")
            row["payload_bytes"] = tr.get("payload_bytes")
        if et == "runtime_state":
            st = ev.get("state") or {}
            row["n_past"] = st.get("n_past")
            row["kv_entries"] = st.get("kv_entries")
            row["hidden_sha256"] = (st.get("hidden_sha256") or "")[:16]
        if et == "token_selected":
            row["token"] = ev.get("token")
        if et == "logits":
            row["argmax"] = ev.get("argmax")
            row["entropy"] = ev.get("entropy")
        rows.append(row)
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace-dir", default="/data/models/traces")
    parser.add_argument("--session", required=True)
    args = parser.parse_args()

    base = Path(args.trace_dir)
    entry = base / f"{args.session}_node-a_entry.jsonl"
    middle = base / f"{args.session}_node-b_middle.jsonl"
    final = base / f"{args.session}_node-c_final.jsonl"

    report = {"session": args.session}

    transport = analyze_transport(args.trace_dir)
    report["transport_memcmp"] = transport
    failed_tcp = [r for r in transport if not r.get("ok")]
    report["transport_all_ok"] = len(failed_tcp) == 0

    for name, path in [("entry", entry), ("middle", middle), ("final", final)]:
        if path.exists():
            report[name] = summarize_trace(load_jsonl(str(path)), name)

    # Cross-stage hidden compare at prefill step 0
    def hidden_at(events: list[dict], step: int = 0, phase: str = "prefill") -> str | None:
        for ev in events:
            if ev.get("event") == "hidden" and ev.get("step") == step and ev.get("phase") == phase:
                return (ev.get("stats") or {}).get("sha256")
        return None

    if entry.exists() and middle.exists():
        e0 = load_jsonl(str(entry))
        m0 = load_jsonl(str(middle))
        he = hidden_at(e0)
        hm = hidden_at(m0)
        report["prefill_hidden_entry_sha256"] = (he or "")[:16]
        report["prefill_hidden_middle_sha256"] = (hm or "")[:16]
        report["prefill_entry_middle_match"] = he == hm

    # First decode step logits argmax on final
    if final.exists():
        for ev in load_jsonl(str(final)):
            if ev.get("event") == "logits" and ev.get("phase") == "decode":
                report["first_decode_logits"] = {
                    "step": ev.get("step"),
                    "argmax": ev.get("argmax"),
                    "entropy": ev.get("entropy"),
                }
                break

    print(json.dumps(report, indent=2))
    return 0 if report.get("transport_all_ok") else 1


if __name__ == "__main__":
    sys.exit(main())
