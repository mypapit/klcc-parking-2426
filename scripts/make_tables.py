"""Build the first useful experiment tables from run metrics."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import _bootstrap  # noqa: F401
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def _parse_run_name(name: str) -> dict:
    parts = name.split("_")
    out = {"run": name}
    if len(parts) >= 2:
        out["depth"] = parts[0]
        out["prep"] = parts[1]
    for part in parts[2:]:
        if part.startswith("L"):
            out["length"] = int(part[1:])
        elif part.startswith("b"):
            out["batch"] = int(part[1:])
        elif part.startswith("s"):
            out["seed"] = int(part[1:])
    return out


def main() -> int:
    tables = ROOT / "outputs" / "tables"
    tables.mkdir(parents=True, exist_ok=True)
    rows = []
    for metrics_path in (ROOT / "outputs" / "runs").glob("*/**/metrics.json"):
        payload = json.loads(metrics_path.read_text())
        payload.update(_parse_run_name(payload.get("run", metrics_path.parent.name)))
        payload["experiment"] = metrics_path.parents[1].name
        payload["path"] = str(metrics_path)
        rows.append(payload)
    if rows:
        df = pd.DataFrame(rows)
        df.to_csv(tables / "T_runs_index.csv", index=False)
        for exp in ("E1", "E2", "E3"):
            sub = df[df["experiment"] == exp].copy()
            if not sub.empty:
                sub.to_csv(tables / f"T_{exp.lower()}_runs.csv", index=False)
    print(f"wrote tables to {tables}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
