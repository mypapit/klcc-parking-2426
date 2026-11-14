import _bootstrap  # noqa: F401
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd  # noqa: E402

raw = pd.read_csv(ROOT.parent / "klcc_parking.csv")
raw = raw[(raw.sitename == "NWD") & (raw.siterename == "KLCC")].copy()
print("FIX 5: zero readings in the whole raw CSV:", int((raw.available == 0).sum()))
post = raw[pd.to_datetime(raw.lastupdate) >= "2022-04-25"]
print("FIX 5: zero readings post 2022-04-25:", int((post.available == 0).sum()))
print("FIX 5: 5300 readings post 2022-04-25:", int((post.available == 5300).sum()))
print()
print("FIX 7: weekend share of days")
from data import split_bounds  # noqa: E402
from utils import load_config  # noqa: E402

cfg = load_config()
series = pd.read_parquet(cfg["data"]["processed"]).set_index("timestamp")["available"].astype(float)
bounds = split_bounds(series, cfg["splits"]["history_end"], cfg["splits"]["test_b_start"], cfg["splits"]["test_b_end"])
ta_days = series.loc[bounds.test_a[0]:bounds.test_a[1]].index.normalize().unique()
weekend = sum(1 for d in ta_days if d.day_of_week >= 5)
print(f"  Test A: {weekend}/{len(ta_days)} = {100*weekend/len(ta_days):.1f}%")
