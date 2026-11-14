"""Record the execution environment and run a GPU smoke test."""
from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
from pathlib import Path

import _bootstrap  # noqa: F401
import numpy as np
import tensorflow as tf

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "env"


def run(cmd: list[str]) -> str:
    return subprocess.check_output(cmd, text=True).strip()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--freeze", action="store_true")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    freeze_path = OUT / "requirements_frozen.txt"
    if args.freeze or not freeze_path.exists():
        freeze_path.write_text(run([sys.executable, "-m", "pip", "freeze"]) + "\n")
    payload = {
        "python": sys.version,
        "platform": platform.platform(),
        "tensorflow": tf.__version__,
        "build_info": tf.sysconfig.get_build_info(),
        "gpus": [d.name for d in tf.config.list_physical_devices("GPU")],
        "freeze_path": str(freeze_path),
        "nvidia_smi": run(["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv"]),
        "cpu": run(["bash", "-lc", "lscpu | grep -E 'Model name|Socket|Core|Thread'"]),
        "memory": run(["bash", "-lc", "free -g | grep Mem"]),
    }
    (OUT / "env.json").write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n")
    x = np.random.rand(64, 18, 1).astype("float32")
    inp = tf.keras.Input((18, 1))
    h = tf.keras.layers.LSTM(8)(inp)
    h = tf.keras.layers.RepeatVector(18)(h)
    h = tf.keras.layers.LSTM(8, return_sequences=True)(h)
    model = tf.keras.Model(inp, h)
    model.compile("adam", "mse")
    hist = model.fit(x, x, epochs=1, batch_size=32, verbose=0)
    payload["smoke_loss"] = float(hist.history["loss"][0])
    (OUT / "env.json").write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
