# Figure notes for the JTEC manuscript figures

These notes record the provenance of every reworked figure so a reviewer (or
the next agent) can trace each panel back to a run, a seed and a table. The
authoritative plan is `docs/superpowers/plans/2026-11-10-jtec-figures.md`,
with the reviewer's corrections in `...-REVIEW.md`.

## Shared style

All reworked figures use `src/figure_style.py`:

- single-column width 3.5 in, double-column width 7.16 in;
- 8 pt font baseline, set AFTER the figure size so matplotlib cannot shrink
  the text to fit;
- Okabe-Ito colour-blind-safe cycle;
- every figure is written as both PDF and 300 dpi PNG in one `save_figure`
  call.

Seed/run metadata: the addendum requires the seed in the file metadata of
single-seed figures. PNG accepts arbitrary text keys, so `Seed`, `Run` and
`Length` are stored there. PDF silently drops unknown keys with a
`UserWarning: Unknown infodict keyword`, so the same information goes into
the standard PDF Info fields `Title`, `Subject` and `Keywords`.

## Per-figure provenance

| Figure | Size (in) | Run / seed | Basis |
|---|---|---|---|
| F01_raw_series | 7.16 x 2.2 | seed-free (raw data) | raw `data/raw/klcc-parking.csv`, 170,781 10-min bins from 1 Jan 2021 to 31 Mar 2024 |
| F03_seasonality_heatmap | 3.5 x 2.3 | seed-free (train split) | mean available bays over the T2b train split, per day of week and 10-min slot |
| F09_error_over_time | 7.16 x 2.2 | E3/D3_P0_L18_b128_s0, seed 0, L=18 | `scores_testA.parquet`, p99.5 threshold 0.003847 from `metrics.json` |
| F10_reconstructions | 3.5 x 3.0 | E3/D3_P0_L18_b128_s0, seed 0, L=18 | four captioned windows, see below |
| F11_threshold_sensitivity_5seeds | 3.5 x 2.4 | E3/D3_P0_L18_b128_s0..s4 | `T8c_threshold_5seeds.csv`, 5-seed mean and std |
| F12_depth_ablation | 3.5 x 2.0 | 5-seed aggregate | `T5_depth_ablation.csv`, D1/D2/D3, seeds 0-4 |
| F13_method_fault_heatmap | 7.16 x 2.6 | 5-seed aggregate (see caveat) | `T7_methods_by_fault.csv`, Test A and Test B panels |

F13 caveat: D1/D2/D3 and B3 are 5-seed aggregates, but B1, B2 and B4 are
single-seed baselines (seed 0 only), and the figure mixes the Test A and
Test B panels. This is recorded here rather than silently resolved; the
addendum did not ask for the baselines to be re-run at five seeds.

## The anomaly score and the threshold

The anomaly score is `mean((z_obs - z_recon)^2)` over the window, where both
terms are in z-score units (the P0 preprocessing standardises to the train
mean/std). This is consistent with the p99.5 threshold of about 0.0038 for
D3/P0/L=18, seed 0. F10's `prepare_window` uses the same formula, and its
model input is reshaped to `(batch, length, 1)` float32 to match
`make_windows`; feeding a 2-D window is accepted silently by Keras but
returns a different reconstruction, which is the bug that first made the
panel scores diverge from the stored scores.

## F10: the four windows, exact dates and scores

All from run E3/D3_P0_L18_b128_s0 (D3/P0, L=18, seed 0), p99.5 threshold
0.003847. Window start indices refer to the Test A series, whose first
timestamp is 2023-06-30 04:40. `prepare_window` takes the window START index;
`make_windows` drops NaN-containing windows, so positional indexing into a
scored array is offset by about 99 positions and must be done by timestamp.

| Panel | Window | Score | Flagged |
|---|---|---|---|
| (a) normal weekday | start_i 4751, the highest-scoring window on 2023-08-02 (all 144 of that day's scores are below the threshold) | 0.00179 | not flagged |
| (b) real anomaly on a verified event day | start_i 13022, window ending 2023-09-28 17:50, during Beautyexpo & Cosmobeaute Malaysia 2023 (27-30 Sep 2023) | 0.05261 | flagged |
| (c) injected F3 full-capacity block | start_i 9548, first segment of copy 0 (`injections/testA_F3_copy0.csv`) | 0.01326 | flagged |
| (d) injected F6 noise burst | start_i 22028, first segment of copy 0 (`injections/testA_F6_copy0.csv`) | 0.01302 | FLAGGED |

Two corrections to the plan, verified against the stored artifacts:

- Panel (b)'s score 0.05261 matches the stored `scores_testA.parquet` value
  0.052595 to 1e-5, confirming the reshape fix.
- The plan claimed panel (d) would score 5.76e-05 and NOT be flagged. That
  was wrong. The correct first-segment F6 score is 0.01302, which is flagged.
  The plan's F3 figure of 0.00398 was the FIRST POSITIVE window at
  `start_i - L + 1` (the first window containing the fault), not the window
  starting at `start_i` that panel (c) shows. Both readings are recorded here
  so the caption does not repeat the error.

## F12 caption basis

F12's means cover F1, F2, F3, F5 and F6 on BOTH test sets, matching T5. F4 is
absent from T5 because the single-copy `fault_metrics.csv` it is built from
recorded zero positive windows for F4 at the time T5 was generated (the F4
injection NaN bug, since fixed). In the corrected `fault_metrics_allcopies.csv`,
F4 is detectable at Test A PR-AUC about 0.92, so its exclusion from F12/T5 is
a stale-data artefact, not a property of the fault. The caption must not imply
that F4 is undetectable. A test (`tests/test_f12_depth.py`) recomputes T5's
D3/P0/L18 mean from the raw run files and pins this basis.

## T8c threshold table and the false-alarm definition

`T8c_threshold_5seeds.csv` (threshold level, macro over faults) is built by
`scripts/make_f11_thresholds.py` by re-scoring the injected series for the
five D3/P0/L18 seeds (inference only, no retraining). Companion tables:
`T8c_threshold_5seeds_byfault.csv` (per fault) and
`T8c_threshold_5seeds_per_run.csv` (raw per-run rows).

At p99.5 the 5-seed macro result is precision 0.198 +/- 0.040, recall
0.597 +/- 0.015, F1 0.290 +/- 0.043, and 26.79 +/- 8.75 flagged windows per
week. The single-run T8b value is 26.04/week (687 flagged windows over
26.38 weeks), which is the number the addendum quotes as "26 false
alarms/week".

Definition: `false_alarms_per_week` is FLAGGED WINDOWS per week on clean
Test A, the same definition `make_f11.py:98` and T8b use. Merged anomaly
events per week is a different, smaller number (3.76 +/- 0.98 at p99.5) and is
reported separately as `merged_false_alarm_events_per_week`. The manuscript
must say which one it means.

Macro-denominator note: the macro average is a NaN-skipping mean over the six
faults, and the tables record the contributing fault count (`n_faults_*`
columns). All six faults contribute at every percentile threshold (95-99.9).
Only the POT family is partial: at the POT thresholds some faults produce no
flagged window at all, so their precision is undefined and only 3 of 6 faults
contribute to the POT precision/F1 macro (recall still covers all 6). The
single-seed `make_f11.py` now uses the same NaN-skipping definition, so the
two scripts agree: 5-seed p99.9 F1 = 0.4131, single-seed = 0.4307. Before this
fix, `make_f11.py` used a NaN-PROpagating mean and reported NaN at p99.9,
p99.95 and for POT precision; that inconsistency is now removed.

## Follow-ups

- The summary generator's figure list at `scripts/make_summary3.py:186-203`
  was updated to the reworked JTEC set and now cites
  `F11_threshold_sensitivity_5seeds.pdf` on the 5-seed macro basis instead of
  the old single-run figure, and the T8c tables were added to the table list.
  `RESULT_SUMMARY3.md` was regenerated from it, so the stale F11 caption is
  resolved. Provenance for each figure is in this file.
- F13's baselines B1/B2/B4 remain single-seed; re-running them at five seeds
  is a separate decision.
