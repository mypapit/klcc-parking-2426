import _bootstrap  # noqa: F401
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from audit import classify_event  # noqa: E402
from data import split_bounds  # noqa: E402
from inject import inject_copy  # noqa: E402
from utils import load_config  # noqa: E402

cfg = load_config()
processed = pd.read_parquet(cfg["data"]["processed"]).set_index("timestamp")
series = processed["available"].astype(float)
bounds = split_bounds(series, cfg["splits"]["history_end"], cfg["splits"]["test_b_start"], cfg["splits"]["test_b_end"])
train = series.loc[bounds.train[0]:bounds.train[1]]
jump_threshold = float(np.percentile(train.diff().abs().dropna(), 99.99))
raw_eval = series.loc[bounds.test_a[0]:bounds.test_a[1]]

injected, positions = inject_copy(raw_eval, "testA", "F4", 0, cfg)
rows = []
for _, seg in positions.iterrows():
    window = injected.loc[pd.Timestamp(seg["start"]):pd.Timestamp(seg["end"])]
    rows.append({
        "value": float(window.iloc[0]),
        "rule": classify_event(window, jump_threshold),
    })
df = pd.DataFrame(rows)
print("F4 injected values and their classification (Test A, copy 0):")
print(df["rule"].value_counts().to_string())
print()
mis = df[df["rule"] != "sensor-suspect"]
print("classified as demand anomaly:", len(mis))
if len(mis):
    print("  their injected values:", sorted(mis["value"].unique()))
print()
print("value distribution overall:")
print(df["value"].describe().to_string())
print("  at 5300:", int((df["value"] == 5300).sum()), " at 0:", int((df["value"] == 0).sum()))
print("  mid-range (0 < v < 5300):", int(((df["value"] > 0) & (df["value"] < 5300)).sum()))
