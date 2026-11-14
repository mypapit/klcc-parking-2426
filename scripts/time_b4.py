"""Time the matrix profile baseline with distinct sizes to avoid JIT cache reuse."""
from __future__ import annotations

import sys
import time
from pathlib import Path

import _bootstrap  # noqa: F401
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from baselines import matrix_profile_score  # noqa: E402
from data import split_bounds  # noqa: E402
from utils import load_config  # noqa: E402


def main() -> int:
    cfg = load_config()
    df = pd.read_parquet(cfg["data"]["processed"]).set_index("timestamp")
    s = df["available"].astype(float)
    b = split_bounds(s, cfg["splits"]["history_end"], cfg["splits"]["test_b_start"], cfg["splits"]["test_b_end"])
    train = s.loc[b.train[0]:b.train[1]].to_numpy()
    test = s.loc[b.test_a[0]:b.test_a[1]].to_numpy()
    for ref_len, eval_len in ((1000, 1000), (2000, 2000), (4000, 4000)):
        start = time.perf_counter()
        matrix_profile_score(train[:ref_len], test[:eval_len], 18, chunk=512)
        elapsed = time.perf_counter() - start
        rate = (ref_len * eval_len) / elapsed
        full = (len(train) * len(test)) / rate
        print(f"ref {ref_len} eval {eval_len} -> {elapsed:.2f}s | full 46569x26612 est {full / 60:.1f} min")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
