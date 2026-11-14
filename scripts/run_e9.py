"""E9: retrain D3/P0 with long full-capacity runs masked from train and val.

Fifteen runs: three window lengths x five seeds, same settings as E2.
Resumable via the existing metrics.json skip logic in run_one.py.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--python", default=str(ROOT.parent / ".venv" / "bin" / "python"))
    parser.add_argument("--min-run", type=int, default=6, help="mask runs of at least this many bins (6 = 1 hour)")
    parser.add_argument("--length", type=int, default=0, help="restrict to one window length (0 = all)")
    parser.add_argument("--seed", type=int, default=-1, help="restrict to one seed (-1 = all)")
    args = parser.parse_args()
    runs = [
        ("D3", "P0", length, 128, seed)
        for length in (6, 18, 36)
        for seed in range(5)
        if (not args.length or length == args.length) and (args.seed < 0 or seed == args.seed)
    ]
    script = ROOT / "scripts" / "run_one.py"
    for depth, prep, length, batch, seed in runs:
        cmd = [
            args.python, str(script),
            "--exp", "E9", "--depth", depth, "--prep", prep,
            "--length", str(length), "--batch", str(batch), "--seed", str(seed),
            "--mask-full-capacity", str(args.min_run),
        ]
        print("RUN", " ".join(cmd), flush=True)
        subprocess.run(cmd, check=True)
    print(json.dumps({"experiment": "E9", "runs": len(runs), "mask_min_run": args.min_run}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
