"""Build RESULT_SUMMARY3.md: the summary after the final checks.

Applies HANDOVER_ADDENDUM_02_final_checks.md:

1. the end-to-end pipeline table is rebuilt from the current artifacts (the old
   one was written before the F4 re-injection), per fault, per length, over five
   seeds, with the confusion matrix broken down per fault type
2. the flag-rate sentence in the contradictions section is replaced with the
   numbers only: flags per week are already normalised by duration, so the
   difference is a real rate difference, not a duration effect
3. the T7 pivot in Key numbers gains a method column and a Test B counterpart
4. F4 is displayed as a single-sample excursion (5 sigma, clamped), because the
   5-sigma magnitude always leaves the valid reading range on this feed
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

CHANGELOG = """# RESULT_SUMMARY3

This is the summary after `HANDOVER_ADDENDUM_02_final_checks.md`. It supersedes
`RESULT_SUMMARY2.md`, which remains in place as the record of the addendum-01
fixes. The F4 root-cause correction (NaN sigma, not an off-by-one in the label
rule) is accepted and carried forward unchanged.

## Changelog by check

| Check | What changed | Tables | Figures |
|---|---|---|---|
| 1. End-to-end pipeline contradicted T7 for F4 | The old pipeline table said 0 of 50 F4 injections were detected by the AE while T7 gave F4 PR-AUC 0.83 to 0.90. The table was written before the F4 re-injection and read the pre-fix NaN injections. It is now rebuilt from the current artifacts for D3/P0 at L = 6, 18, 36 over five seeds: injections, detected at p99.5, detection rate, and the sensor/demand classification of the detected ones. The injected-classifier confusion matrix is broken down per fault type. | T_check1_pipeline, T_check1_pipeline_per_run, T_check1_confusion_by_fault | |
| 2. Flag-rate sentence was wrong | The contradictions section called the flag-rate gap a duration effect, but flags per week are already normalised by duration. The sentence is replaced with the numbers only: 26.77 against 10.41 flagged per week at L=18, and the event-day share p0 = 0.216 on Test A against 0.099 on Test B. No interpretation. | contradictions section 4 | |
| 3. T7 rows unlabelled | The methods-against-fault-types pivot in Key numbers had no method column, so the seven rows were anonymous. The method column is added, and the same pivot is produced for Test B as well as Test A. | T7 pivots in Key numbers | |
| 4. F4 renamed | F4 is a 5-sigma excursion from the value at the injection point. On this feed 5 sigma is about 4,966 bays, so the spike always leaves the valid reading range and is clamped to 0 or 5300. It is therefore a single-sample boundary excursion, not a mid-range spike, and is displayed as such in tables, figures and text. The magnitude is unchanged. | all tables with a fault column | F13, F14 |

### A finding that came out of Check 1

Rebuilding the pipeline exposed something the earlier tables could not show. Of
the 50 F4 injections on Test A, 44 clamp to a boundary and 6 land mid-range
(because the source value was already near full capacity). On Test B the split
is 30 and 20. The Section 7 rules only recognise a value as sensor-suspect when
it is at or below 0, at or above 5300, flat over six bins, or a large jump, so
the mid-range injections are classified as demand anomaly. That count is a
property of the clamping, not a defect in the classifier, and it is the reason
F4 is named an excursion rather than a spike.

### E9

E9 (retrain without full-capacity runs) is now run and reported. The hypothesis
was that the 2.86 percent full-capacity training points explain why F1 (stuck
sensor) sits near chance. They do not: F1 changes by less than 0.004 PR-AUC at
every length on both test sets, while F4 falls by 0.20 to 0.24 and F5 by 0.04.
The mask removes the model's main exposure to boundary values, which makes the
single-sample excursion harder to rank rather than easier.

On the operational measure, event recall at p99.5, the damage is far larger.
Across all faults on Test A, recall falls from 0.865 to 0.549 at L=6, from
0.849 to 0.580 at L=18 and from 0.871 to 0.624 at L=36. F4 goes from 1.000
detected at every length in E2 to 0.304 to 0.604 in E9, and F1 at L=6 goes
from 0.400 to never. E9 is a negative result and is reported as such in
section 5.
"""


def md(df: pd.DataFrame, limit: int | None = None, index: bool = False) -> str:
    if limit is not None and len(df) > limit:
        return df.head(limit).to_markdown(index=index) + f"\n\n_{len(df) - limit} further rows in the CSV file._"
    return df.to_markdown(index=index)


def main() -> int:
    lines: list[str] = [CHANGELOG, ""]

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
            sub = t7[(t7["length"] == 18) & (t7["testset"] == testset)]
            if not sub.empty:
                lines += [f"### Methods against fault types (T7), {testset} at L=18", ""]
                pivot = sub.pivot_table(index="method", columns="fault", values="pr_auc_mean")
                pivot = pivot.rename(columns={c: fault_label(c) for c in pivot.columns})
                # Check 3: the method column must be visible in the output, so
                # the pivot's index is rendered rather than dropped
                lines += [md(pivot.round(4), index=True), ""]
        lines += ["All methods use the 5-copy injection protocol. The baselines cover Test A and Test B; every method has saved per-timestamp scores.", ""]

    check1 = pd.read_csv(TABLES / "T_check1_pipeline.csv") if (TABLES / "T_check1_pipeline.csv").exists() else None
    if check1 is not None:
        lines += ["### Check 1: end-to-end pipeline per fault, D3/P0, all five seeds", ""]
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

    t11 = pd.read_csv(TABLES / "T11_base_rate_summary.csv") if (TABLES / "T11_base_rate_summary.csv").exists() else None
    if t11 is not None:
        lines += ["### Day-level base-rate test (T11), the main test", ""]
        keep = ["length", "split", "day_event_exact_n_days_mean", "day_event_exact_n_anomaly_days_mean", "day_event_exact_p0_mean", "day_event_exact_k_mean", "day_event_exact_binomial_p_mean", "day_event_exact_permutation_p_mean"]
        keep = [c for c in keep if c in t11.columns]
        lines += [md(t11[keep].round(5)), ""]
        weekend = [c for c in t11.columns if "weekend" in c and c.endswith("_mean")]
        if weekend:
            lines += ["Weekend arm (Fix 7): " + ", ".join(f"{c[:-5]} = {t11[c].iloc[0]:.4f}" for c in weekend), ""]

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

    lines += ["## 2. Tables", ""]
    for filename, title in (
        ("T_check1_pipeline.csv", "Check 1: end-to-end pipeline per fault"),
        ("T_check1_pipeline_per_run.csv", "Check 1: end-to-end pipeline per run"),
        ("T_check1_confusion_by_fault.csv", "Check 1: injected-classifier confusion per fault"),
        ("T_e9_vs_e2_pr_auc.csv", "E9 against E2: PR-AUC per fault"),
        ("T_e9_vs_e2.csv", "E9 against E2: merged events on Test A"),
        ("T_e9_vs_e2_event_recall.csv", "E9 against E2: event recall per fault"),
        ("T11_base_rate_summary_E9.csv", "E9 day-level base-rate summary"),
        ("T_full_capacity_readings.csv", "Fix 5: full-capacity readings per split"),
        ("T_full_capacity_monthly.csv", "Fix 5: full-capacity readings per month"),
        ("T5_depth_ablation.csv", "T5 Depth ablation"),
        ("T5b_event_recall_by_fault.csv", "T5b Event recall by fault"),
        ("T7_methods_by_fault.csv", "T7 Methods x fault types"),
        ("T8_threshold_summary.csv", "T8 Threshold sensitivity"),
        ("T8b_threshold_sensitivity_detail.csv", "T8b Threshold sensitivity detail"),
        ("T8c_threshold_5seeds.csv", "T8c Threshold sensitivity, 5-seed macro over all six faults"),
        ("T8c_threshold_5seeds_byfault.csv", "T8c Threshold sensitivity per fault, 5 seeds"),
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
        if "fault" in frame.columns:
            frame = frame.copy()
            frame["fault"] = frame["fault"].map(lambda f: fault_label(f) if isinstance(f, str) and f in {"F1", "F2", "F3", "F4", "F5", "F6"} else f)
        lines += [f"### {title}", "", md(frame, limit=limit), ""]

    lines += ["## 3. Figures", ""]
    # the reworked JTEC set; provenance for each figure is in FIGURE_NOTES.md
    for filename, description in (
        ("F01_raw_series.pdf", "Raw 10-min available bays, Jan 2021 - Mar 2024, with the COVID-19 and inconsistent-sensor exclusion bands and the T2b split strip"),
        ("F02_data_quality.pdf", "Sampling interval, duplicate rows and full-capacity readings per month"),
        ("F03_seasonality_heatmap.pdf", "Weekly seasonality heatmap over the train split, Monday at top, single column"),
        ("F07_learning_curves.pdf", "Learning curves as mean +/- std over seeds per window length"),
        ("F09_error_over_time.pdf", "Anomaly score over clean Test A on a log y axis, verified KLCC event bands, merged events, p99.5 threshold (E3/D3_P0_L18_b128_s0)"),
        ("F10_reconstructions.pdf", "Four captioned observed/reconstruction panels in bays with score and flagged state (E3/D3_P0_L18_b128_s0): normal weekday, real event-day anomaly, injected F3, injected F6"),
        ("F11_threshold_sensitivity_5seeds.pdf", "Threshold sensitivity at 5 seeds: flagged windows per week on clean Test A, and injected precision/recall/F1 macro-averaged over all six faults (T8c)"),
        ("F12_depth_ablation.pdf", "Depth ablation: mean PR-AUC and mean F1 at p99.5, std over 5 seeds, caption basis F1/F2/F3/F5/F6 on both test sets (T5, single-copy)"),
        ("F13_method_fault_heatmap_testA.pdf", "Method x fault PR-AUC, Test A, three panels L=6/18/36 with a shared colour scale; D1/D2/D3/B3 5-seed, B1/B2/B4 single-seed"),
        ("F13_method_fault_heatmap_testB.pdf", "Method x fault PR-AUC, Test B, same layout and seed caveat as Test A"),
        ("F14_pr_curves.pdf", "Precision-recall curves per fault type"),
        ("F16_audit_breakdown.pdf", "Audit breakdown by category per window length"),
        ("F17_permutation_test.pdf", "Day-level event test: null distribution with the observed value"),
        ("F18_fault_classification_confusion.pdf", "Section 7 classifier confusion matrix"),
    ):
        if (OUT / "figures" / filename).exists():
            lines.append(f"- `{filename}`: {description} (source: outputs/figures/{filename})")
    lines.append("")

    lines += ["## 4. Results that contradict or qualify the original claims", ""]
    contradictions = OUT / "CONTRADICTIONS3.md"
    lines += [contradictions.read_text().strip() if contradictions.exists() else "Not yet compiled.", ""]

    lines += ["## 5. E9: retrain without full-capacity runs", ""]
    e9 = OUT / "E9_REPORT.md"
    lines += [e9.read_text().strip() if e9.exists() else "E9 not yet run.", ""]

    lines += ["## 6. ASSUMPTIONS, ISSUES and open questions", ""]
    for name in ("ASSUMPTIONS2.md", "ISSUES2.md", "OPEN_QUESTIONS2.md"):
        path = OUT / name
        if path.exists():
            lines += [f"### {name.replace('.md', '')}", "", path.read_text().strip(), ""]

    target = OUT / "RESULT_SUMMARY3.md"
    target.write_text("\n".join(lines) + "\n")
    print(f"wrote {target} ({len(lines)} blocks)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
