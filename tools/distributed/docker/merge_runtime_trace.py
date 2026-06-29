#!/usr/bin/env python3
"""Merge distributed runtime JSONL traces into a unified timeline."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def load_events(path: Path) -> list[dict]:
    events: list[dict] = []
    if not path.is_file():
        return events
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            events.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return events


def main() -> int:
    parser = argparse.ArgumentParser(description="Merge runtime trace JSONL files")
    parser.add_argument("traces", nargs="+", help="Trace JSONL files or directories")
    parser.add_argument("-o", "--output", help="Write merged timeline JSON")
    args = parser.parse_args()

    all_events: list[dict] = []
    for raw in args.traces:
        p = Path(raw)
        if p.is_dir():
            for f in sorted(p.glob("*.jsonl")):
                all_events.extend(load_events(f))
        else:
            all_events.extend(load_events(p))

    all_events.sort(key=lambda e: (e.get("step", -1), e.get("ts_ms", 0), e.get("worker", "")))

    steps: dict[int, list[dict]] = {}
    for ev in all_events:
        step = int(ev.get("step", -1))
        steps.setdefault(step, []).append(ev)

    timeline: list[dict] = []
    for step in sorted(steps):
        chain = steps[step]
        entry = next((e for e in chain if e.get("event") == "hidden" and e.get("worker") == "entry"), None)
        middle = next((e for e in chain if e.get("event") == "hidden" and e.get("worker") == "middle"), None)
        final_h = next((e for e in chain if e.get("event") == "hidden" and e.get("worker") == "final"), None)
        logits = next((e for e in chain if e.get("event") == "logits"), None)
        token = next((e for e in chain if e.get("event") == "token_selected"), None)
        timeline.append({
            "step": step,
            "phase": chain[0].get("phase") if chain else "",
            "events": chain,
            "summary": {
                "entry_hidden_sha256": (entry or {}).get("stats", {}).get("sha256"),
                "middle_hidden_sha256": (middle or {}).get("stats", {}).get("sha256"),
                "final_hidden_sha256": (final_h or {}).get("stats", {}).get("sha256"),
                "logits_argmax": (logits or {}).get("argmax"),
                "logits_sha256": (logits or {}).get("stats", {}).get("sha256"),
                "token": (token or {}).get("token"),
            },
        })

    out = {"timeline": timeline, "event_count": len(all_events)}
    text = json.dumps(out, indent=2)
    if args.output:
        Path(args.output).write_text(text)
    else:
        print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
