"""Build the external reference data used by the audit.

- public_holidays.csv from the holidays package (KL subdiv=14)
- weather_hourly.csv from the Open-Meteo archive API (rain plausibility for Section 7)
- festive_periods.csv from the same holidays package
- school_holidays.csv is left empty: it needs a KPM source URL per row
- klcc_events.csv is left empty: the handover forbids guessing events

Nothing here is guessed. Where a source cannot be reached or a value cannot be
verified, the file is written empty and the gap is recorded in outputs/ISSUES.md.
"""
from __future__ import annotations

import csv
import json
import sys
import urllib.request
from pathlib import Path

import _bootstrap  # noqa: F401
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from events import write_public_holidays  # noqa: E402
from utils import ensure_dir  # noqa: E402

OUT = ROOT / "data" / "external"

WEATHER_URL = (
    "https://archive-api.open-meteo.com/v1/archive"
    "?latitude=3.1579&longitude=101.7123"
    "&start_date=2022-04-25&end_date=2024-03-31"
    "&hourly=precipitation"
    "&timezone=Asia%2FKuala_Lumpur"
)


def build_weather() -> None:
    target = OUT / "weather_hourly.csv"
    try:
        with urllib.request.urlopen(WEATHER_URL, timeout=120) as response:
            payload = json.loads(response.read())
    except Exception as exc:  # noqa: BLE001
        print(f"weather fetch failed: {exc}")
        pd.DataFrame(columns=["timestamp", "precipitation_mm", "source_url"]).to_csv(target, index=False)
        return
    hourly = payload["hourly"]
    df = pd.DataFrame({
        "timestamp": pd.to_datetime(hourly["time"]),
        "precipitation_mm": hourly["precipitation"],
    })
    df["source_url"] = WEATHER_URL
    df.to_csv(target, index=False)
    print(f"weather rows: {len(df)}")


def build_festive() -> None:
    target = OUT / "festive_periods.csv"
    try:
        import holidays
    except ImportError:
        print("holidays package missing; festive_periods.csv not written")
        return
    frames = []
    for year in (2022, 2023, 2024):
        cal = holidays.country_holidays("MY", subdiv="14", years=[year])
        for day, name in cal.items():
            frames.append({"date": day, "occasion": name, "year": year})
    if not frames:
        pd.DataFrame(columns=["date", "occasion", "source_url"]).to_csv(target, index=False)
        return
    df = pd.DataFrame(frames).sort_values("date")
    df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
    df["source_url"] = "https://pypi.org/project/holidays/ (MY subdiv=14)"
    # occasion names contain commas, so quote them explicitly
    df.to_csv(target, index=False, quoting=csv.QUOTE_MINIMAL)
    print(f"festive rows: {len(df)}")


def main() -> int:
    ensure_dir(OUT)
    df = write_public_holidays(OUT / "public_holidays.csv")
    print(f"public holidays: {len(df)}")
    build_weather()
    build_festive()
    school = OUT / "school_holidays.csv"
    if not school.exists():
        pd.DataFrame(columns=["date_start", "date_end", "source_url"]).to_csv(school, index=False)
        print("school_holidays.csv written empty: needs a KPM source URL per row")
    events = OUT / "klcc_events.csv"
    if not events.exists():
        pd.DataFrame(columns=["date_start", "date_end", "event_name", "venue", "source_url"]).to_csv(events, index=False)
        print("klcc_events.csv written empty: every row needs a verifiable source URL")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
