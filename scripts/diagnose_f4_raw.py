import _bootstrap  # noqa: F401
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from data import split_bounds  # noqa: E402
from inject import inject_copy  # noqa: E402
from metrics import event_labels  # noqa: E402
from utils import load_config  # noqa: E402
from windows import make_windows  # noqa: E402

cfg = load_config()
length = 18
series = pd.read_parquet(cfg["data"]["processed"]).set_index("timestamp")["available"].astype(float)
bounds = split_bounds(series, cfg["splits"]["history_end"], cfg["splits"]["test_b_start"], cfg["splits"]["test_b_end"])
test_a = series.loc[bounds.test_a[0]:bounds.test_a[1]]

injected_raw, positions = inject_copy(test_a, "testA", "F4", 0, cfg)
x, ts = make_windows(injected_raw.to_numpy(), injected_raw.index, length)
y = event_labels(ts, positions, length)
print(f"RAW (no preprocess): windows={len(x)}, positives={int(y.sum())}")

# is it the label range or the dropped windows?
dropped = 0
for _, seg in positions.iterrows():
    lo = pd.Timestamp(seg["start"])
    hi = pd.Timestamp(seg["end"]) + pd.Timedelta(minutes=10 * (length - 1))
    inside = ((ts >= lo) & (ts <= hi)).sum()
    if inside == 0:
        dropped += 1
print(f"segments with zero windows in range: {dropped}/{len(positions)}")

# check the injection itself: does the raw value change at the segment?
for i, seg in positions.head(3).iterrows():
    lo = pd.Timestamp(seg["start"])
    print(f"seg {i}: raw before={test_a.loc[lo]:.1f}, after={injected_raw.loc[lo]:.1f}")
