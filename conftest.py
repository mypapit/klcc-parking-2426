"""Pytest bootstrap.

The Hermes runtime exports PYTHONPATH to its own Python 3.14 site-packages. When
that leaks into this project's Python 3.12 venv, the 3.14 numpy shadows the venv's
numpy 2.5.2 and every pandas import fails with a compiled-extension mismatch.

conftest.py is imported before test collection, so the cleanup runs once for the
whole session, before any test module imports numpy or pandas.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
for extra in (ROOT / "src", ROOT / "scripts"):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

import _bootstrap  # noqa: E402,F401
