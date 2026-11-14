"""Interpreter bootstrap for this project.

Import this before any third-party package (numpy, pandas, TensorFlow).

The Hermes runtime exports PYTHONPATH pointing at its own Python 3.14
site-packages. When that leaks into this project's Python 3.12 venv, the 3.14
numpy shadows the venv's numpy 2.5.2 and every pandas import fails with a
compiled-extension mismatch ("No module named 'numpy._core._multiarray_umath'").

site.py adds PYTHONPATH entries to sys.path after .pth files are processed, so
this cannot be fixed from a .pth file. Clearing it here, before any third-party
import, is the only reliable point.
"""
from __future__ import annotations

import os
import sys

_MARKER = f"python{sys.version_info.major}.{sys.version_info.minor}"


def _is_foreign(path: str) -> bool:
    """A site-packages directory belonging to a different Python version."""
    return "site-packages" in path and "python3." in path and _MARKER not in path


def apply() -> None:
    raw = os.environ.get("PYTHONPATH", "")
    if raw:
        parts = [p for p in raw.split(os.pathsep) if p]
        keep = [p for p in parts if not _is_foreign(p)]
        if keep != parts:
            if keep:
                os.environ["PYTHONPATH"] = os.pathsep.join(keep)
            else:
                os.environ.pop("PYTHONPATH", None)
    sys.path[:] = [p for p in sys.path if not _is_foreign(p)]
    # the foreign tree also ships pytest plugins built for another Python, and
    # those load through entry points rather than sys.path. Blocking autoload
    # keeps this project's test run independent of the caller's environment.
    os.environ.setdefault("PYTEST_DISABLE_PLUGIN_AUTOLOAD", "1")


apply()
