import _bootstrap  # noqa: F401
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
t7 = pd.read_csv(ROOT / "outputs" / "tables" / "T7_methods_by_fault.csv")
print("=== which protocol does each method use? ===")
for m in sorted(t7.method.unique()):
    if m.startswith("D"):
        runs = sorted((ROOT / "outputs" / "runs" / "E3").glob("D3_*")) + sorted((ROOT / "outputs" / "runs" / "E2").glob("D3_*"))
    else:
        runs = sorted((ROOT / "outputs" / "runs" / "E4").glob(f"{m}_*"))
    n_all = sum(1 for r in runs if (r / "fault_metrics_allcopies.csv").exists())
    print(f"{m}: {len(runs)} runs, {n_all} with 5-copy metrics")

print()
print("=== D3 L=18 Test A, mean PR-AUC per fault ===")
d3 = t7[(t7.method == "D3") & (t7.testset == "testA") & (t7.length == 18)]
print(d3.groupby("fault")["pr_auc_mean"].mean().round(4).to_string())
