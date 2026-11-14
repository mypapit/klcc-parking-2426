"""Experiment runner."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp", required=True, choices=["E1", "E2", "E3", "E4"])
    parser.add_argument("--python", default=str(ROOT.parent / ".venv" / "bin" / "python"))
    args = parser.parse_args()
    runs = []
    if args.exp == "E3":
        runs = [("D3", prep, length, 128, seed) for prep in ("P0", "P1", "P2") for length in (6, 18, 36) for seed in range(5)]
    elif args.exp == "E2":
        runs = [(depth, "P0", length, 128, seed) for depth in ("D1", "D2", "D3") for length in (6, 18, 36) for seed in range(5)]
    elif args.exp == "E1":
        runs = [("D3", "P1", length, batch, seed) for length in (6, 18, 36) for batch in (32, 128) for seed in range(5)]
    else:
        raise SystemExit("E4 baselines are run separately")
    script = ROOT / "scripts" / "run_one.py"
    for depth, prep, length, batch, seed in runs:
        cmd = [args.python, str(script), "--exp", args.exp, "--depth", depth, "--prep", prep, "--length", str(length), "--batch", str(batch), "--seed", str(seed)]
        print("RUN", " ".join(cmd), flush=True)
        subprocess.run(cmd, check=True)
    print(json.dumps({"experiment": args.exp, "runs": len(runs)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
