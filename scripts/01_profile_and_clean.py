"""Profile and clean the raw KLCC feed into a 10-minute parquet."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import _bootstrap  # noqa: F401
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from utils import load_config  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", default=None)
    args = parser.parse_args()
    cfg = load_config()
    raw = Path(args.raw or cfg["data"]["raw"])
    out = Path(cfg["data"]["processed"])
    out.parent.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(raw)
    df["lastupdate"] = pd.to_datetime(df["lastupdate"])
    df = df[df["siteid"] == cfg["data"]["siteid"]].sort_values(["lastupdate", "rowid"]).copy()
    counts = [{"step": "load_filter_sort", "rows_before": len(df), "rows_after": len(df), "rows_removed": 0}]
    before = len(df)
    df = df.drop_duplicates()
    counts.append({"step": "drop_exact_duplicates", "rows_before": before, "rows_after": len(df), "rows_removed": before - len(df)})
    before = len(df)
    df = df[~df.duplicated("lastupdate", keep="last")]
    counts.append({"step": "same_timestamp_keep_last", "rows_before": before, "rows_after": len(df), "rows_removed": before - len(df)})
    before = len(df)
    df = df[(df["available"] >= 0) & (df["available"] <= df["totalbay"])]
    counts.append({"step": "drop_out_of_range", "rows_before": before, "rows_after": len(df), "rows_removed": before - len(df)})
    keep_for_profile = df.copy()
    df = df[df["lastupdate"] >= pd.Timestamp(cfg["data"]["exclusion_start"])].copy()
    counts.append({"step": "exclude_pre_2022_04_25", "rows_before": before, "rows_after": len(df), "rows_removed": before - len(df)})
    grid = df.set_index("lastupdate")["available"].resample(cfg["data"]["freq"]).last().to_frame()
    grid["interpolated_flag"] = grid["available"].isna()
    gap_mask = grid["available"].isna()
    gap_id = (~gap_mask).cumsum()
    gap_lengths = gap_mask.groupby(gap_id).sum()
    grid["available"] = grid["available"].interpolate(limit=cfg["data"]["interpolation_max_bins"], limit_area="inside")
    grid["interpolated_flag"] = grid["available"].notna() & gap_mask
    grid.loc[grid["available"].isna(), "interpolated_flag"] = False
    grid = grid.reset_index().rename(columns={"lastupdate": "timestamp"})
    grid.to_parquet(out, index=False)
    tables = ROOT / "outputs" / "tables"
    tables.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(counts).to_csv(tables / "T2_data_accounting.csv", index=False)
    profile = pd.DataFrame({
        "rows_site": [len(df)],
        "raw_rows": [len(keep_for_profile)],
        "output": [str(out)],
    })
    profile.to_csv(tables / "T2_profile.csv", index=False)
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
