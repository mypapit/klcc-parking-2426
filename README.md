# KLCC Parking Occupancy 2022-2024

Dataset for the reproducibility of an article submitted to the **Journal of
Telecommunication, Electronic and Computer Engineering** (ISSN: 2180-1843,
eISSN: 2289-8131), http://jtec.utem.edu.my

**Dataset name:** KLCC Parking Occupancy 2022-2024
**License:** Creative Commons Attribution-ShareAlike 4.0 International (CC BY-SA 4.0)

## Journal Article

**Title:** Temporal Anomaly Detection in Vehicle Parking Occupancy Data Using
LSTM Autoencoder and Rule-Based Fault Screening

**Authors:**

1. Mohammad Hafiz bin Ismail (corresponding author)
   Faculty of Computer and Mathematical Sciences, Universiti Teknologi MARA,
   Arau, 02600, Malaysia
2. Tajul Rosli Razak
   Faculty of Computer and Mathematical Sciences, Universiti Teknologi MARA,
   Shah Alam, 40450, Malaysia

**Corresponding author contact:**

- Email: mohammadhafiz@uitm.edu.my
- Web: https://mypapit.net

## Purpose

The repository contains the parking occupancy time series, the derived
external-event and calendar series, the experiment configuration, the source
code for preprocessing, fault injection, LSTM autoencoder training, rule-based
fault screening, baselines and evaluation, plus the generated tables and
figures. Anyone should be able to regenerate every number and figure reported
in the article from the material here.

## Repository layout

    config/
      default.yaml              Experiment configuration: data paths, splits,
                                window lengths, preprocessing variants,
                                model depths, seeds, threshold percentiles.

    data/
      raw/klcc-parking.csv      Raw 10-minute occupancy readings for site 7
                                (KLCC, total capacity 5300 bays), columns:
                                rowid, siteid, sitename, available,
                                lastupdate, totalbay, siterename.
                                166,578 rows, 2021-01-01 to 2024-03-31.
      processed/
        klcc_10min.parquet      Cleaned and regridded 10-minute series
                                (timestamp, available, interpolated_flag).
                                101,807 rows, 2022-04-25 to 2024-03-31.
      external/
        klcc_events.csv         Event calendar used for the event-recall test.
        public_holidays.csv     Malaysian public holidays.
        school_holidays.csv     School holiday periods.
        festive_periods.csv     Festive periods used by the rule-based screen.
        weather_hourly.csv      Hourly weather covariates.

    src/                        Library modules: data loading, masking,
                                windowing, fault injection, models, thresholds,
                                metrics, audit, plots.
    scripts/                    Numbered and named entry points that reproduce
                                each stage (environment check, cleaning,
                                aggregation, experiments, audits, tables,
                                figures).
    tests/                      Pytest suite covering data integrity, windowing,
                                thresholds, audit and figure generation.
    outputs/                    Generated logs, runs, tables and figures
                                (regenerable, not required as input).

    LICENSE                     Verbatim CC BY-SA 4.0 legal code.
    CITATION.cff                Machine-readable citation metadata (CFF 1.1.0).

## Data splits

The configuration fixes the evaluation protocol:

- Training and validation history: up to 2023-12-31 23:59:59.
- Test A: held-out portion of the history, used for in-distribution scoring.
- Test B: 2024-01-01 to 2024-03-31, used as the out-of-time evaluation.
- The period up to 2022-04-25 is excluded from the cleaned series.
- Gaps of up to 3 bins (30 minutes) are linearly interpolated and flagged in
  `interpolated_flag`.

## Environment

Recorded in `outputs/env/env.json` and pinned in
`outputs/env/requirements_frozen.txt`:

- Linux, Python 3.12, TensorFlow 2.20.0 (CUDA 12.5, cuDNN 9)
- Intel Core i7-14700, 64 GB RAM, NVIDIA GeForce RTX 4060 Ti 16 GB

## Reproducing the results

    python scripts/00_env_check.py --freeze          # record environment
    python scripts/01_profile_and_clean.py           # raw -> processed series
    python scripts/build_external.py                 # external series build
    python scripts/aggregate.py                      # aggregation checks
    python scripts/run_experiments.py                # autoencoder grid (E1-E3)
    bash    scripts/run_all_baselines_v2.sh          # baselines B1-B4
    python scripts/run_audit_all.py                  # audit and base-rate tests
    python scripts/make_tables.py                    # tables T1-T13
    python scripts/make_figures.py                   # figures F01-F14
    pytest                                           # verification suite

Run every command from the repository root with the pinned virtual
environment. Paths in `config/default.yaml` are absolute and must be adjusted
to the local checkout before running.

## Notes

- All model runs use seeds 0 to 4 where the article reports five-seed
  aggregates; single-seed baselines are marked as such in
  `outputs/FIGURE_NOTES.md` and the table files.
- `outputs/FIGURE_NOTES.md` records the provenance of each figure: run
  identifier, seed, window length and the source table.
- The raw occupancy file is a snapshot of a public parking availability feed
  for the KLCC site, redistributed unmodified under the license in this
  README.

## License

The dataset **KLCC Parking Occupancy 2022-2024**, together with the code and
derived material in this repository, is released under the **Creative Commons
Attribution-ShareAlike 4.0 International License (CC BY-SA 4.0)**:
https://creativecommons.org/licenses/by-sa/4.0/

The verbatim legal code is in the [`LICENSE`](LICENSE) file and governs.

Researchers are welcome to use, reuse and redistribute the dataset, including
for commercial purposes, provided that:

1. **Attribution is given to both works.** Any publication, product or derived
   dataset that uses this material must cite BOTH
   - the journal article: Mohammad Hafiz bin Ismail and Tajul Rosli Razak,
     "Temporal Anomaly Detection in Vehicle Parking Occupancy Data Using LSTM
     Autoencoder and Rule-Based Fault Screening", *Journal of
     Telecommunication, Electronic and Computer Engineering*; and
   - the dataset: Mohammad Hafiz bin Ismail and Tajul Rosli Razak, "KLCC
     Parking Occupancy 2022-2024", licensed under CC BY-SA 4.0.

   Both works must be attributed to their respective authors in full, with the
   article title, dataset title, author names, journal name and the license
   notice.
2. **ShareAlike is honoured.** If you remix, transform or build upon this
   dataset, you must distribute your contributions under the same CC BY-SA 4.0
   license.
3. **Changes are indicated.** State clearly if you modified the data or code.

Suggested citation (also in [`CITATION.cff`](CITATION.cff), which GitHub, Zenodo
and reference managers read automatically):

    Ismail, M. H., & Razak, T. R. (2026). KLCC Parking Occupancy 2022-2024
    (Version 1.0.0) [Data set]. Licensed under CC BY-SA 4.0.
    https://mypapit.net

Contact the corresponding author (mohammadhafiz@uitm.edu.my) for the article's
full bibliographic details once it is published.
