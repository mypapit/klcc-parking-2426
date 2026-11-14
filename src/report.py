"""Report assembly helpers."""
from __future__ import annotations

from pathlib import Path


def append_markdown(path: str | Path, title: str, body: str) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a") as fh:
        fh.write(f"\n## {title}\n\n{body}\n")
