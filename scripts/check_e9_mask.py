import _bootstrap  # noqa: F401
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from data import split_bounds  # noqa: E402
from utils import load_config  # noqa: E402
from windows import make_windows  # noqa: E402

cfg = load_config()
series = pd.read_parquet(cfg["data"]["processed"]).set_index("timestamp")["available"].astype(float)
bounds = split_bounds(series, cfg["splits"]["history_end"], cfg["splits"]["test_b_start"], cfg["splits"]["test_b_end"])


def mask_full_capacity(values: pd.Series, min_run: int = 6) -> pd.Series:
    out = values.copy()
    run_id = (values != values.shift(1)).cumsum()
    for _, group in values.groupby(run_id):
        if group.iloc[0] == 5300 and len(group) >= min_run:
            out.loc[group.index] = np.nan
    return out


for name, span in (("train", bounds.train), ("val", bounds.val)):
    original = series.loc[span[0]:span[1]]
    masked = mask_full_capacity(original)
    removed_points = int(masked.isna().sum() - original.isna().sum())
    print(f"{name}: {len(original)} points, {removed_points} removed ({removed_points/len(original)*100:.2f}%)")
    for length in (6, 18, 36):
        orig_windows, _ = make_windows(original.to_numpy(), original.index, length)
        masked_windows, _ = make_windows(masked.to_numpy(), masked.index, length)
        removed = len(orig_windows) - len(masked_windows)
        print(f"  L={length}: {len(orig_windows)} -> {len(masked_windows)} windows ({removed} removed, {removed/len(orig_windows)*100:.2f}%)")
