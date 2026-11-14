"""Assemble the final results summary from available outputs."""
from __future__ import annotations

import json
from pathlib import Path

import _bootstrap  # noqa: F401
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"

TABLE_ORDER = [
    ("T1_dataset_excerpt.csv", "T1 Dataset excerpt"),
    ("T2_data_accounting.csv", "T2 Data accounting"),
    ("T2b_splits.csv", "T2b Split boundaries and window counts"),
    ("T2_profile.csv", "T2 Profile"),
    ("T2_data_quality.csv", "T2 Data quality over time"),
    ("T3_implementation_details.csv", "T3 Implementation details"),
    ("T3b_params_per_layer.csv", "T3b Parameters per layer"),
    ("T_e3_summary.csv", "T6 Preprocessing ablation (E3, mean +/- std over 5 seeds)"),
    ("T5_depth_ablation.csv", "T5 Depth ablation (E2, mean +/- std over 5 seeds)"),
    ("T_e1_summary.csv", "T4 Replication grid (E1, mean +/- std over 5 seeds)"),
    ("T7_methods_by_fault.csv", "T7 Methods x fault types"),
    ("T8_threshold_summary.csv", "T8 Threshold sensitivity (mean +/- std over 5 seeds)"),
    ("T11_base_rate_tests.csv", "T11 Base-rate tests per run"),
    ("T11_base_rate_summary.csv", "T11 Base-rate summary per window length"),
    ("T12_computational_cost.csv", "T12 Computational cost"),
    ("T12_cost_summary.csv", "T12 Cost summary"),
    ("T13_testA_vs_testB_summary.csv", "T13 Test A against Test B (out-of-time)"),
    ("T10_klcc_events.csv", "T10 Verified KLCC event archive with detections inside each window"),
    ("T_fault_classification_summary.csv", "Section 7 classifier: true class against predicted class"),
    ("T9_anomaly_audit.csv", "T9 Anomaly audit (per event)"),
]

FIGURE_ORDER = [
    ("F01_raw_series.pdf", "Full 10-minute series with split boundaries and excluded periods"),
    ("F02_data_quality.pdf", "Median sampling interval per day and duplicate rows per month"),
    ("F03_seasonality_heatmap.pdf", "Mean available bays by day-of-week and time-of-day, train only"),
    ("F04_mstl_decomposition.pdf", "MSTL decomposition of one representative month"),
    ("F05_preprocessing_comparison.pdf", "Raw series against P1 and P2 residuals over one week"),
    ("F07_learning_curves.pdf", "Validation loss per epoch for every E3 run"),
    ("F08_error_distribution.pdf", "Train and Test A reconstruction-error distributions with threshold lines"),
    ("F09_error_over_time.pdf", "Test A error over time with threshold line and flagged events shaded"),
    ("F10_reconstructions.pdf", "Observed against reconstructed windows"),
    ("F11_threshold_sensitivity.pdf", "Flagged-window count against threshold percentile"),
    ("F12_depth_ablation.pdf", "Mean injected PR-AUC per depth and window length"),
    ("F13_prep_pr_auc.pdf", "Mean injected PR-AUC per preprocessing variant and window length"),
    ("F13_method_fault_heatmap_testA.pdf", "PR-AUC by method and fault type on Test A"),
    ("F13_method_fault_heatmap_testB.pdf", "PR-AUC by method and fault type on Test B"),
    ("F16_audit_breakdown.pdf", "Merged anomaly events by audit category per window length"),
    ("F17_permutation_test.pdf", "Permutation p-value against holiday-match fraction per run"),
    ("F14_pr_curves.pdf", "Precision-recall curves per fault type"),
    ("F15_detection_delay.pdf", "Detection delay per injected fault type"),
    ("F18_fault_classification_confusion.pdf", "True fault class against predicted class"),
    ("F19_cost_vs_performance.pdf", "Trainable parameters against mean PR-AUC"),
]


def perm_p(value: float) -> str:
    """Permutation p is bounded below by the number of permutations used."""
    if pd.isna(value):
        return "n/a"
    if value <= 0.0:
        return "< 0.0005 (2000 permutations)"
    return f"{value:.4f}"


def md_table(df: pd.DataFrame, max_rows: int | None = None) -> str:
    if max_rows is not None and len(df) > max_rows:
        shown = df.head(max_rows)
        note = f"\n\n_{len(df) - max_rows} further rows in the CSV file._"
        return shown.to_markdown(index=False) + note
    return df.to_markdown(index=False)


def key_numbers() -> list[str]:
    lines: list[str] = []
    t6 = OUT / "tables" / "T_e3_summary.csv"
    if t6.exists():
        df = pd.read_csv(t6)
        best = df.loc[df["mean_pr_auc_mean"].idxmax()]
        lines.append(f"- Best preprocessing and window length: {best['prep']} at L={int(best['length'])}, mean injected PR-AUC {best['mean_pr_auc_mean']:.3f} +/- {best['mean_pr_auc_std']:.3f} (source: outputs/tables/T_e3_summary.csv)")
        lines.append("  - This is the mean over the 5 seeds of each run's mean PR-AUC over the injected faults. It is the per-run summary statistic for the preprocessing ablation, not the same as T7's fault-level means, which pool the full 5-copy protocol across seeds.")
        pivot = df.pivot_table(index="length", columns="prep", values="mean_pr_auc_mean")
        lines.append("- Mean PR-AUC by preprocessing and window length:")
        for length, row in pivot.iterrows():
            lines.append(f"  - L={int(length)}: " + ", ".join(f"{k} {v:.3f}" for k, v in row.items()) + " (source: outputs/tables/T_e3_summary.csv)")
    t5 = OUT / "tables" / "T5_depth_ablation.csv"
    if t5.exists():
        df = pd.read_csv(t5)
        lines.append("- Depth ablation, mean injected PR-AUC:")
        for _, row in df.iterrows():
            lines.append(f"  - {row['depth']} L={int(row['length'])}: {row['mean_pr_auc_mean']:.3f} +/- {row['mean_pr_auc_std']:.3f} (source: outputs/tables/T5_depth_ablation.csv)")
    t11 = OUT / "tables" / "T11_base_rate_summary.csv"
    if t11.exists():
        df = pd.read_csv(t11)
        for _, row in df.iterrows():
            lines.append(f"- Base-rate test at L={int(row['length'])}: {row['events_mean']:.1f} merged events, {row['matched_exact_mean']:.1f} on special days (p0 {row['p0_exact_mean']:.4f}), binomial p {row['binomial_p_exact_mean']:.3g}, permutation p {perm_p(row['permutation_p_exact_mean'])}")
            lines.append(f"  - public holiday alone: p0 {row['p0_holiday_mean']:.4f}, {row['matched_holiday_mean']:.1f} matched, permutation p {perm_p(row['permutation_p_holiday_mean'])}")
            lines.append(f"  - KLCC event day alone: p0 {row['p0_event_mean']:.4f}, {row['matched_event_mean']:.1f} matched, permutation p {perm_p(row['permutation_p_event_mean'])}")
        cat_cols = [c for c in df.columns if c.startswith("cat_") and c.endswith("_mean")]
        for _, row in df.iterrows():
            parts = [f"{c[4:-5].replace('_', ' ')} {row[c]:.1f}" for c in cat_cols if row[c] > 0]
            if parts:
                lines.append(f"- Audit categories at L={int(row['length'])}: " + "; ".join(parts))
    t13 = OUT / "tables" / "T13_testA_vs_testB_summary.csv"
    if t13.exists():
        df = pd.read_csv(t13)
        lines.append("- Out-of-time comparison, mean injected PR-AUC:")
        for length in sorted(df["length"].unique()):
            sub = df[df["length"] == length].set_index("testset")
            if {"testA", "testB"} <= set(sub.index):
                lines.append(f"  - L={int(length)}: Test A {sub.loc['testA', 'mean_pr_auc_mean']:.3f}, Test B {sub.loc['testB', 'mean_pr_auc_mean']:.3f} (source: outputs/tables/T13_testA_vs_testB_summary.csv)")
    t7 = OUT / "tables" / "T7_methods_by_fault.csv"
    if t7.exists():
        df = pd.read_csv(t7)
        d3 = df[(df["method"] == "D3") & (df["testset"] == "testA")]
        if not d3.empty:
            lines.append("- D3 injected PR-AUC on Test A by fault and window length (mean over 5 seeds):")
            for _, row in d3.sort_values(["length", "fault"]).iterrows():
                lines.append(f"  - L={int(row['length'])} {row['fault']}: {row['pr_auc_mean']:.3f} +/- {row['pr_auc_std']:.3f} (source: outputs/tables/T7_methods_by_fault.csv)")
        # every method against every fault on Test A at L=18
        methods = [m for m in ("D1", "D2", "D3", "B1", "B2", "B3", "B4") if m in set(df["method"])]
        sub = df[(df["length"] == 18) & (df["testset"] == "testA") & (df["method"].isin(methods))]
        if not sub.empty:
            pivot = sub.pivot_table(index="method", columns="fault", values="pr_auc_mean")
            lines.append("- All methods against all fault types, Test A at L=18 (mean PR-AUC):")
            for method in methods:
                if method not in pivot.index:
                    continue
                parts = [f"{fault} {pivot.loc[method, fault]:.3f}" for fault in pivot.columns if pd.notna(pivot.loc[method, fault])]
                lines.append(f"  - {method}: " + ", ".join(parts))
            lines.append("  - The neural runs use the full 5-copy injection protocol; the baselines use a single copy, because the baseline runner does not persist scores or models. The baselines also cover Test A only.")
    return lines


def main() -> int:
    lines = ["# RESULTS_SUMMARY", ""]
    lines += ["## 1. Environment", ""]
    env_path = OUT / "env" / "env.json"
    if env_path.exists():
        lines += ["```json", env_path.read_text().strip(), "```", ""]
    else:
        lines += ["Not recorded.", ""]

    lines += ["## 2. Data accounting", ""]
    for filename, title in TABLE_ORDER[:2]:
        path = OUT / "tables" / filename
        if path.exists():
            lines += [f"### {title}", "", md_table(pd.read_csv(path)), ""]

    lines += ["### External reference data", ""]
    ext_rows = []
    for key, label, filename in (
        ("holidays", "Public holidays (MY subdiv 14)", "public_holidays.csv"),
        ("festive", "Festive periods", "festive_periods.csv"),
        ("events", "KLCC events", "klcc_events.csv"),
        ("school", "School holidays", "school_holidays.csv"),
        ("weather", "Hourly precipitation", "weather_hourly.csv"),
    ):
        path = ROOT / "data" / "external" / filename
        if path.exists():
            frame = pd.read_csv(path)
            ext_rows.append({"reference": label, "file": filename, "rows": len(frame), "status": "populated"})
        else:
            ext_rows.append({"reference": label, "file": filename, "rows": 0, "status": "missing"})
    ext = pd.DataFrame(ext_rows)
    lines += [md_table(ext), ""]
    lines += [
        "Every row of the KLCC event archive carries a source URL that was fetched "
        "during collection; no event date is inferred from a recurrence pattern. The "
        "school-holiday calendar is empty because each row needs a KPM source, which "
        "is a decision for the owner (see OPEN_QUESTIONS.md).",
        "",
    ]

    lines += ["## 3. Tables", ""]
    for filename, title in TABLE_ORDER[2:]:
        path = OUT / "tables" / filename
        if path.exists():
            # the per-event audit has one row per detection; show a sample and the counts
            limit = 25 if filename == "T9_anomaly_audit.csv" else 60
            lines += [f"### {title}", "", md_table(pd.read_csv(path), max_rows=limit), ""]

    lines += ["## 4. Figures", ""]
    for filename, description in FIGURE_ORDER:
        path = OUT / "figures" / filename
        if path.exists():
            lines.append(f"- `{filename}`: {description} (source: outputs/figures/{filename})")
    lines.append("")

    lines += ["## 5. Key numbers", ""]
    lines += key_numbers()
    lines.append("")

    lines += ["## 6. Results that contradict or qualify the original paper's claims", ""]
    contradictions = OUT / "CONTRADICTIONS.md"
    lines += [contradictions.read_text().strip() if contradictions.exists() else "Not yet compiled.", ""]

    lines += ["## 7. ASSUMPTIONS and ISSUES", ""]
    lines += ["### ASSUMPTIONS", "", (OUT / "ASSUMPTIONS.md").read_text().strip() if (OUT / "ASSUMPTIONS.md").exists() else "None.", ""]
    lines += ["### ISSUES", "", (OUT / "ISSUES.md").read_text().strip() if (OUT / "ISSUES.md").exists() else "None.", ""]

    lines += ["## 8. Open questions for the owner", ""]
    questions = OUT / "OPEN_QUESTIONS.md"
    lines += [questions.read_text().strip() if questions.exists() else "None recorded.", ""]

    (OUT / "RESULTS_SUMMARY.md").write_text("\n".join(lines))
    print(OUT / "RESULTS_SUMMARY.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
