#!/usr/bin/env python3
"""Analyze monolithic vs distributed traces and report first divergence."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mono-trace", required=True)
    parser.add_argument("--dist-entry", required=True)
    parser.add_argument("--dist-middle", default="")
    parser.add_argument("--dist-final", default="")
    parser.add_argument("--verify-bin", default="verify_decode_loop_parity")
    args = parser.parse_args()

    cmd = [
        args.verify_bin,
        "--mono-trace", args.mono_trace,
        "--dist-entry", args.dist_entry,
    ]
    if args.dist_middle:
        cmd.extend(["--dist-middle", args.dist_middle])
    if args.dist_final:
        cmd.extend(["--dist-final", args.dist_final])

    proc = subprocess.run(cmd, capture_output=True, text=True)
    stdout = proc.stdout.strip()
    if stdout:
        try:
            report = json.loads(stdout)
        except json.JSONDecodeError:
            print(stdout)
            return proc.returncode
        print(json.dumps(report, indent=2))
        if report.get("status") == "divergence":
            print("\n" + report.get("root_cause", ""), file=sys.stderr)
        return proc.returncode

    print(proc.stderr or "analyze failed", file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())
