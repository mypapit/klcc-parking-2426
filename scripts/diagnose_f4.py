import _bootstrap  # noqa: F401
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from data import split_bounds  # noqa: E402
from features import Preprocessor  # noqa: E402
from inject import inject_copy  # noqa: E402
from metrics import event_labels  # noqa: E402
from utils import load_config  # noqa: E402
from windows import make_windows  # noqa: E402

cfg = load_config()
length = 18
series = pd.read_parquet(cfg["data"]["processed"]).set_index("timestamp")["available"].astype(float)
bounds = split_bounds(series, cfg["splits"]["history_end"], cfg["splits"]["test_b_start"], cfg["splits"]["test_b_end"])
test_a = series.loc[bounds.test_a[0]:bounds.test_a[1]]

for prep in ("P0", "P2"):
    pre = Preprocessor(prep)
    pre.window_length = length
    pre.fit(series.loc[bounds.train[0]:bounds.train[1]])
    injected_raw, positions = inject_copy(test_a, "testA", "F4", 0, cfg)
    injected = pre.transform(injected_raw)
    x, ts = make_windows(injected.to_numpy(), injected.index, length)
    y = event_labels(ts, positions, length)
    print(f"prep={prep}: windows={len(x)}, positives={int(y.sum())}")
    if int(y.sum()) == 0:
        print("  positions head:")
        print(positions[["start", "end"]].head(3).to_string(index=False))
        print("  ts range:", ts.min(), "->", ts.max())
        print("  first pos start:", positions.iloc[0]["start"], " end:", positions.iloc[0]["end"])
        # count how many window ends fall in [start, end + (L-1)*10min]
        seg = positions.iloc[0]
        lo = pd.Timestamp(seg["start"])
        hi = pd.Timestamp(seg["end"]) + pd.Timedelta(minutes=10 * (length - 1))
        inside = ((ts >= lo) & (ts <= hi)).sum()
        print(f"  windows with end in [lo, hi] = {inside}")
        # what does the injected series actually look like there?
        print("  injected values at seg:", injected.loc[lo:hi].to_numpy()[:5], "...")
        print("  original values at seg:", test_a.loc[lo:lo + pd.Timedelta(minutes=20)].to_numpy()[:5], "...")
        # is the window start index such that the spike is inside the window?
        print("  ts around lo:", [t for t in ts if lo - pd.Timedelta(minutes=20) <= t <= lo + pd.Timedelta(minutes=30)][:5])
