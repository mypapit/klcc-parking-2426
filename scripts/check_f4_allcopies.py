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
sigma = float(np.nanstd(train.to_numpy()))
print(f"train sigma = {sigma:.4f}; 5 sigma = {5 * sigma:.4f}")
print()

for testset, span in (("testA", bounds.test_a), ("testB", bounds.test_b)):
    raw_eval = series.loc[span[0]:span[1]]
    rows = []
    for copy_id in range(5):
        injected, positions = inject_copy(raw_eval, testset, "F4", copy_id, cfg)
        for _, seg in positions.iterrows():
            window = injected.loc[pd.Timestamp(seg["start"]):pd.Timestamp(seg["end"])]
            rows.append({
                "copy": copy_id,
                "source": float(raw_eval.loc[pd.Timestamp(seg["start"])]),
                "injected": float(window.iloc[0]),
                "rule": classify_event(window, jump_threshold),
            })
    df = pd.DataFrame(rows)
    print(f"=== {testset}: {len(df)} injections ===")
    print(df.groupby("rule")["injected"].describe().to_string())
    print()
    mis = df[df["rule"] != "sensor-suspect"]
    print(f"misclassified: {len(mis)}")
    if len(mis):
        print("  injected value range:", mis["injected"].min(), "to", mis["injected"].max())
        print("  source value range:", mis["source"].min(), "to", mis["source"].max())
        print("  first 5:")
        print(mis.head(5).to_string(index=False))
    print()
