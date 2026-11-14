"""Build RESULT_SUMMARY2.md: the summary after the addendum fixes.

Structure (addendum, Deliverable section):
1. a changelog at the top listing which tables and figures changed per fix
2. T7, T9, T10, T11, T13, the classifier table and F02, F07, F11, F13, F14,
   F16, F17, F18 regenerated
3. the new Fix 5 table
4. the contradictions section updated (no Test B "easier" claim, day-level test
   results, F4 results)
"""
from __future__ import annotations

import sys
from pathlib import Path

import _bootstrap  # noqa: F401
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from labels import fault_label  # noqa: E402

OUT = ROOT / "outputs"
TABLES = OUT / "tables"

CHANGELOG = """# RESULT_SUMMARY2

This is the post-fix summary, produced after `HANDOVER_ADDENDUM_01_fixes.md`.
It supersedes `RESULTS_SUMMARY.md`, which remains in place as the pre-fix record.

## Changelog by fix

| Fix | What changed | Tables | Figures |
|---|---|---|---|
| 1. F4 label bug | Root cause was **not** the label rule. `inject_copy` computed `sigma_train = float(np.std(series.to_numpy()))` over a series containing NaN, so `np.std` returned NaN and every F4 spike was written as NaN. The affected windows were then dropped by the NaN guard in `make_windows`, so F4 had zero positive windows on the neural path. `src/inject.py` now uses a finite sigma from the train split, clamps the spike into the valid reading range, and refuses injection positions that land on missing data. The label rule is unchanged and correct: a 1-step segment at s covers the L window ends s..s+L-1. | T7, T5, T5b | F13, F14 |
| 2. Classifier truth labels | All six injected faults are sensor faults, so F5 and F6 are relabelled from `demand anomaly` to `sensor fault`. The demand ground truth is now real merged events on verified KLCC event days. Both the rule accuracy and the end-to-end pipeline (detected by the AE and then classified) are reported. | classifier table, T18 | F18 |
| 3. Test A vs Test B | PR-AUC is prevalence-dependent and Test B has about half the windows with the same injections, so the cross-set comparison now leads with ROC-AUC, reports the prevalence, and gives PR-AUC normalised by prevalence. Flag rates per week are added. The "Test B is easier" claim is removed. | T13, T13 flag rates | |
| 4. Day-level base rate | The day-level test is now the main test in T11: the unit is the calendar day, the null redraws which days are anomaly days, 10,000 permutations, seeded. The event-level test is kept and labelled as secondary. Test B is included. | T11 | F17 |
| 5. Full-capacity readings | There are no zero readings in this feed. All 3,181 post-2022-04-25 extremes are 5300 (full-capacity) readings, which the out-of-range rule never flagged. The new table gives counts, run counts and length distributions per split, and the training split is 2.86 percent full-capacity points. | Fix 5 table, monthly table | F02 (extra panel) |
| 6. Baseline protocol | B1 to B4 now save per-timestamp scores, run the 5-copy protocol on Test A and Test B, and B3 runs with five seeds. B4 stays CPU-only and its injected profile is computed exactly (verified against a full AB-join). B4's threshold comes from a cross-half AB-join of the train split, not the self-join, which is dominated by trivial self-matches. The single-copy results are archived. | T7, T12 | F13, F14 |
| 7. Weekend is not an explanation | The weekend category is merged into `unexplained`, with a separate `unexplained_on_weekend` count, and weekend gets its own arm in the day-level test. | T9, T11 | F16 |
| 8. Small corrections | T10 lists only events inside the scored periods; T3 reports the environment as clean values (62 GiB, RTX 4060 Ti 16 GB, Ubuntu 24.04, driver version); F07 plots the mean +/- std band across seeds; F11 gains the injected-F1 right axis and the false-alarms-per-week series; F17 shows the null distribution with the observed value. | T3, T3b, T10 | F02, F07, F11, F17 |

### Counterpoint on Fix 1

The addendum attributes the F4 failure to an off-by-one in the label rule and
proposes a shared label function with the range `[start, end_inclusive + L - 1]`.
That range is already what `event_labels` implements, and the unit test added
here confirms it: a 1-step segment at s with L = 18 gives exactly 18 positive
window ends.

The real defect is one line upstream in the injection. `inject_copy` computed

    sigma_train = float(np.std(series.to_numpy()))

on the evaluation series, which contains NaN gaps, so `np.std` returned NaN.
Only F4 consumes that value, so every F4 spike was written as `value +/- NaN`,
i.e. NaN. `make_windows` drops any window containing NaN, so the fault never
reached the scorer and F4 scored zero positive windows on the neural path. B2
appeared correct only because its own code path replaces NaN with 0.0 before
scoring, which is why its F4 counts looked right (60, 180, 360 = 10 injections
x L) while the neural and B1/B3/B4 paths were all zero.

The consequence is that "re-score F4 from saved scores" is not possible on its
own: the archived scores were computed on the corrupted injection. Every fault
has been re-injected and re-scored instead (inference only, no retraining),
because the position rejection for missing data also changes which windows the
other faults land in.
"""


def read(name: str) -> str | None:
    path = TABLES / name
    if not path.exists():
        return None
    return path.read_text()


def md(df: pd.DataFrame, limit: int | None = None) -> str:
    if limit is not None and len(df) > limit:
        return df.head(limit).to_markdown(index=False) + f"\n\n_{len(df) - limit} further rows in the CSV file._"
    return df.to_markdown(index=False)


def main() -> int:
    lines: list[str] = [CHANGELOG, ""]

    # key numbers
    lines += ["## 1. Key numbers", ""]
    t5 = pd.read_csv(TABLES / "T5_depth_ablation.csv") if (TABLES / "T5_depth_ablation.csv").exists() else None
    if t5 is not None:
        lines += ["### Depth ablation (T5), mean over five seeds", ""]
        cols = ["depth", "length", "mean_pr_auc_mean", "mean_roc_auc_mean", "mean_f1_mean", "event_recall_mean"]
        cols = [c for c in cols if c in t5.columns]
        lines += [md(t5[cols].round(4)), ""]

    t7 = pd.read_csv(TABLES / "T7_methods_by_fault.csv") if (TABLES / "T7_methods_by_fault.csv").exists() else None
    if t7 is not None:
        for testset in ("testA", "testB"):
            lines += [f"### Methods against fault types (T7), {testset} at L=18", ""]
            sub = t7[(t7["length"] == 18) & (t7["testset"] == testset)]
            if not sub.empty:
                pivot = sub.pivot_table(index="method", columns="fault", values="pr_auc_mean")
                pivot = pivot.rename(columns={c: fault_label(c) for c in pivot.columns})
                lines += [md(pivot.round(4)), ""]
        lines += ["All methods use the 5-copy injection protocol. The baselines cover Test A and Test B; every method has saved per-timestamp scores.", ""]

    t11 = pd.read_csv(TABLES / "T11_base_rate_summary.csv") if (TABLES / "T11_base_rate_summary.csv").exists() else None
    if t11 is not None:
        lines += ["### Day-level base-rate test (T11), the main test", ""]
        keep = ["length", "split", "day_event_exact_n_days_mean", "day_event_exact_n_anomaly_days_mean", "day_event_exact_p0_mean", "day_event_exact_k_mean", "day_event_exact_binomial_p_mean", "day_event_exact_permutation_p_mean"]
        keep = [c for c in keep if c in t11.columns]
        lines += [md(t11[keep].round(5)), ""]
        weekend = [c for c in t11.columns if "weekend" in c and c.endswith("_mean")]
        if weekend:
            lines += ["Weekend arm (Fix 7): " + ", ".join(f"{c[:-5]} = {t11[c].iloc[0]:.4f}" for c in weekend), ""]
        eventlevel = [c for c in t11.columns if "eventlevel" in c and c.endswith("_mean")]
        if eventlevel:
            lines += ["Event-level tests (secondary, 2,000 permutations): " + ", ".join(f"{c[:-5]} = {t11[c].iloc[0]:.4f}" for c in eventlevel), ""]

    t13 = pd.read_csv(TABLES / "T13_testA_vs_testB_summary.csv") if (TABLES / "T13_testA_vs_testB_summary.csv").exists() else None
    if t13 is not None:
        lines += ["### Test A against Test B (T13), prevalence-aware", ""]
        keep = [c for c in t13.columns if c.endswith("_mean") or c in ("length", "split")]
        lines += [md(t13[keep].round(5)), ""]

    full = pd.read_csv(TABLES / "T_full_capacity_readings.csv") if (TABLES / "T_full_capacity_readings.csv").exists() else None
    if full is not None:
        lines += ["### Full-capacity readings per split (Fix 5)", ""]
        lines += [md(full), ""]

    cls = pd.read_csv(TABLES / "T_fault_classification_summary.csv") if (TABLES / "T_fault_classification_summary.csv").exists() else None
    if cls is not None:
        lines += ["### Section 7 classifier (Fix 2)", ""]
        lines += [md(cls), ""]

    pipe = pd.read_csv(TABLES / "T_fault_pipeline.csv") if (TABLES / "T_fault_pipeline.csv").exists() else None
    if pipe is not None:
        lines += ["### End-to-end pipeline: detected by the AE and then classified (Fix 2)", ""]
        lines += [md(pipe), ""]

    check1 = pd.read_csv(TABLES / "T_check1_pipeline.csv") if (TABLES / "T_check1_pipeline.csv").exists() else None
    if check1 is not None:
        lines += ["### Check 1: end-to-end pipeline per fault, D3/P0, mean over five seeds", ""]
        lines += [md(check1), ""]
        lines += [
            "`detection_rate` is the share of injected segments the autoencoder flagged at the p99.5 threshold. "
            "`detected_sensor` and `detected_demand` are the Section 7 classification of the detected injections only.",
            "",
        ]
        conf = pd.read_csv(TABLES / "T_check1_confusion_wide.csv") if (TABLES / "T_check1_confusion_wide.csv").exists() else None
        if conf is not None:
            lines += ["#### Injected-classifier confusion matrix broken down per fault type", ""]
            lines += [md(conf), ""]

    # tables
    lines += ["## 2. Tables", ""]
    for filename, title in (
        ("T_full_capacity_readings.csv", "Fix 5: full-capacity readings per split"),
        ("T_full_capacity_monthly.csv", "Fix 5: full-capacity readings per month"),
        ("T5_depth_ablation.csv", "T5 Depth ablation"),
        ("T5b_event_recall_by_fault.csv", "T5b Event recall by fault"),
        ("T7_methods_by_fault.csv", "T7 Methods x fault types"),
        ("T8_threshold_summary.csv", "T8 Threshold sensitivity"),
        ("T8b_threshold_sensitivity_detail.csv", "T8b Threshold sensitivity detail"),
        ("T9_anomaly_audit.csv", "T9 Anomaly audit per event"),
        ("T10_klcc_events.csv", "T10 KLCC events inside the scored periods"),
        ("T11_base_rate_tests.csv", "T11 Base-rate tests per run"),
        ("T11_base_rate_summary.csv", "T11 Base-rate summary per window length"),
        ("T12_cost_summary.csv", "T12 Computational cost, per method and window length"),
        ("T12_computational_cost.csv", "T12 Computational cost, per run"),
        ("T13_testA_vs_testB_summary.csv", "T13 Test A against Test B"),
        ("T13_flag_rates.csv", "T13 Flag rates per week"),
        ("T_fault_classification_summary.csv", "Section 7 classifier summary"),
        ("T_fault_pipeline.csv", "Section 7 end-to-end pipeline"),
        ("T18_confusion_matrix.csv", "T18 Confusion matrix"),
        ("T3_implementation_details.csv", "T3 Implementation details"),
        ("T3b_params_per_layer.csv", "T3b Parameters per layer"),
    ):
        path = TABLES / filename
        if not path.exists():
            continue
        frame = pd.read_csv(path)
        limit = 25 if filename in ("T9_anomaly_audit.csv", "T11_base_rate_tests.csv", "T12_computational_cost.csv") else None
        # Check 4: the fault code F4 is a single-sample excursion, not a
        # mid-range spike, so the display label says so wherever a fault column
        # is rendered. The CSV keeps the code.
        if "fault" in frame.columns:
            frame = frame.copy()
            frame["fault"] = frame["fault"].map(lambda f: fault_label(f) if isinstance(f, str) and f in {"F1", "F2", "F3", "F4", "F5", "F6"} else f)
        lines += [f"### {title}", "", md(frame, limit=limit), ""]

    # figures
    lines += ["## 3. Figures", ""]
    for filename, description in (
        ("F02_data_quality.pdf", "Sampling interval, duplicate rows and full-capacity readings per month"),
        ("F07_learning_curves.pdf", "Learning curves as mean +/- std over seeds per window length"),
        ("F11_threshold_sensitivity.pdf", "Threshold sensitivity: merged events, false alarms per week, injected F1"),
        ("F13_method_fault_heatmap_testA.pdf", "Method x fault PR-AUC, Test A, shared colour scale"),
        ("F13_method_fault_heatmap_testB.pdf", "Method x fault PR-AUC, Test B, shared colour scale"),
        ("F14_pr_curves.pdf", "Precision-recall curves per fault type"),
        ("F16_audit_breakdown.pdf", "Audit breakdown by category per window length"),
        ("F17_permutation_test.pdf", "Day-level event test: null distribution with the observed value"),
        ("F18_fault_classification_confusion.pdf", "Section 7 classifier confusion matrix"),
    ):
        if (OUT / "figures" / filename).exists():
            lines.append(f"- `{filename}`: {description} (source: outputs/figures/{filename})")
    lines.append("")

    # contradictions
    lines += ["## 4. Results that contradict or qualify the original claims", ""]
    contradictions = OUT / "CONTRADICTIONS2.md"
    lines += [contradictions.read_text().strip() if contradictions.exists() else "Not yet compiled.", ""]

    lines += ["## 5. ASSUMPTIONS, ISSUES and open questions", ""]
    for name in ("ASSUMPTIONS2.md", "ISSUES2.md", "OPEN_QUESTIONS2.md"):
        path = OUT / name
        if path.exists():
            lines += [f"### {name.replace('.md', '')}", "", path.read_text().strip(), ""]

    target = OUT / "RESULT_SUMMARY2.md"
    target.write_text("\n".join(lines) + "\n")
    print(f"wrote {target} ({len(lines)} blocks)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
