"""Shared utilities: config loading, seeding, logging, timing."""
from __future__ import annotations

import json
import os
import random
import time
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = REPO_ROOT / "config" / "default.yaml"


def repo_path(*parts: str) -> Path:
    return REPO_ROOT.joinpath(*parts)


def load_config(path: str | Path | None = None) -> dict:
    cfg_path = Path(path) if path else DEFAULT_CONFIG
    with cfg_path.open() as fh:
        return yaml.safe_load(fh)


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def write_json(path: str | Path, payload: dict) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True, default=str)


def set_all_seeds(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)


class Timer:
    def __init__(self) -> None:
        self.start = time.perf_counter()
        self.elapsed = None

    def stop(self) -> float:
        self.elapsed = time.perf_counter() - self.start
        return self.elapsed


@contextmanager
def timer():
    t = Timer()
    yield t
    t.stop()
