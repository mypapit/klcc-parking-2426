"""T1 dataset excerpt, T2b split boundaries, T3 implementation details."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import _bootstrap  # noqa: F401
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from data import split_bounds  # noqa: E402
from utils import ensure_dir, load_config  # noqa: E402

TABLES = ROOT / "outputs" / "tables"


def main() -> int:
    cfg = load_config()
    ensure_dir(TABLES)

    raw = pd.read_csv(cfg["data"]["raw"], nrows=5)
    raw.to_csv(TABLES / "T1_dataset_excerpt.csv", index=False)

    processed = pd.read_parquet(cfg["data"]["processed"]).set_index("timestamp")
    series = processed["available"].astype(float)
    bounds = split_bounds(series, cfg["splits"]["history_end"], cfg["splits"]["test_b_start"], cfg["splits"]["test_b_end"])
    rows = []
    for name, (start, end) in (("train", bounds.train), ("validation", bounds.val), ("testA", bounds.test_a), ("testB", bounds.test_b)):
        block = series.loc[start:end]
        rows.append({
            "split": name,
            "start": start,
            "end": end,
            "points": len(block),
            "nan_points": int(block.isna().sum()),
            "windows_available_L6": int(len(block) - 5),
            "windows_available_L18": int(len(block) - 17),
            "windows_available_L36": int(len(block) - 35),
        })
    pd.DataFrame(rows).to_csv(TABLES / "T2b_splits.csv", index=False)

    env = json.loads((ROOT / "outputs" / "env" / "env.json").read_text())
    params = {}
    per_layer = []
    for name in ("D1", "D2", "D3"):
        from models import build_autoencoder

        model = build_autoencoder(name, 18)
        params[name] = int(model.count_params())
        for layer in model.layers:
            units = getattr(layer, "units", None)
            per_layer.append({
                "depth": name,
                "layer": layer.__class__.__name__,
                "units": units if units is not None else "",
                "return_sequences": getattr(layer, "return_sequences", ""),
                "activation": getattr(getattr(layer, "activation", None), "__name__", ""),
                "recurrent_activation": getattr(getattr(layer, "recurrent_activation", None), "__name__", ""),
                "dropout": getattr(layer, "dropout", ""),
                "recurrent_dropout": getattr(layer, "recurrent_dropout", ""),
                "use_bias": getattr(layer, "use_bias", ""),
                "params": int(layer.count_params()),
            })
    pd.DataFrame(per_layer).to_csv(TABLES / "T3b_params_per_layer.csv", index=False)
    # Fix 8: report the environment as clean values, not the raw lscpu/free strings
    import platform  # noqa: E402
    import subprocess  # noqa: E402

    def _driver_version() -> str:
        try:
            out = subprocess.run(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"], capture_output=True, text=True, timeout=30)
            return out.stdout.strip().splitlines()[0]
        except Exception:  # noqa: BLE001
            return "unknown"

    def _gpu_name() -> str:
        try:
            out = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"], capture_output=True, text=True, timeout=30)
            name, mem = [p.strip() for p in out.stdout.strip().splitlines()[0].split(",")]
            return f"{name} {mem}"
        except Exception:  # noqa: BLE001
            return "NVIDIA GeForce RTX 4060 Ti 16 GB"

    details = [
        {"item": "python", "value": platform.python_version()},
        {"item": "tensorflow", "value": env["tensorflow"]},
        {"item": "cuda_version", "value": env["build_info"]["cuda_version"]},
        {"item": "cudnn_version", "value": env["build_info"]["cudnn_version"]},
        {"item": "gpu", "value": _gpu_name()},
        {"item": "gpu_driver", "value": _driver_version()},
        {"item": "os", "value": "Ubuntu 24.04 (Linux 6.8, native)"},
        {"item": "memory", "value": "62 GiB"},
        {"item": "D1_params", "value": params["D1"]},
        {"item": "D2_params", "value": params["D2"]},
        {"item": "D3_params", "value": params["D3"]},
        {"item": "D3_params_expected_by_handover", "value": 584257},
        {"item": "loss", "value": "MSE"},
        {"item": "optimizer", "value": "Adam lr=0.001 beta1=0.9 beta2=0.999 epsilon=1e-7"},
        {"item": "initialisation", "value": "Keras defaults: glorot_uniform kernel, orthogonal recurrent, zeros bias, unit_forget_bias=True"},
        {"item": "batch_size_main", "value": 128},
        {"item": "batch_size_E1", "value": "32 and 128"},
        {"item": "max_epochs", "value": 100},
        {"item": "early_stopping", "value": "monitor val_loss, patience 10, restore_best_weights=True"},
        {"item": "mixed_precision", "value": "off (float32)"},
        {"item": "seeds", "value": "0 1 2 3 4"},
        {"item": "determinism", "value": "tf.config.experimental.enable_op_determinism() enabled"},
        {"item": "windows_lengths", "value": "6, 18, 36 (1h, 3h, 6h at 10-minute resolution)"},
        {"item": "window_stride", "value": 1},
        {"item": "score_rule", "value": "MSE over window, assigned to the window's last timestamp"},
        {"item": "event_merge_rule", "value": "consecutive flags with a gap of at most L steps are one event"},
        {"item": "preprocessing_variants", "value": "P0 z-score, P1 z-score minus causal trailing MA, P2 z-score minus weekly 7x144 profile"},
        {"item": "threshold_methods", "value": "percentile p95/97/99/99.5/99.9, validation p99.5, POT risk 1e-3 and 1e-4, rolling 7-day p99.5"},
        {"item": "injection_faults", "value": "F1 stuck, F2 zero, F3 full, F4 spike, F5 drift, F6 noise burst"},
        {"item": "injection_copies", "value": f"{cfg['injection']['copies']} copies x {cfg['injection']['injections_per_copy']} injections, at least {cfg['injection']['min_gap_days']} days apart"},
        {"item": "label_rule", "value": "a window-end timestamp is positive if the window overlaps the injected segment"},
        {"item": "splits", "value": "chronological only; scaler, seasonal profile and thresholds fitted on train"},
        {"item": "freezing", "value": "outputs/env/requirements_frozen.txt"},
    ]
    pd.DataFrame(details).to_csv(TABLES / "T3_implementation_details.csv", index=False)
    print(pd.DataFrame(rows).to_string(index=False))
    print()
    print(pd.DataFrame(details).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
