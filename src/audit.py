"""Event audit: Section 7 classifier and Section 8 external validation."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
EXTERNAL = ROOT / "data" / "external"


def load_external() -> dict:
    """Load the external reference tables. Missing files yield empty frames."""
    out = {}
    specs = {
        "holidays": ("public_holidays.csv", {"date"}),
        "events": ("klcc_events.csv", {"date_start", "date_end"}),
        "school": ("school_holidays.csv", {"date_start", "date_end"}),
        "festive": ("festive_periods.csv", {"date"}),
        "weather": ("weather_hourly.csv", {"timestamp"}),
    }
    for key, (filename, required) in specs.items():
        path = EXTERNAL / filename
        if not path.exists():
            out[key] = pd.DataFrame(columns=sorted(required))
            continue
        df = pd.read_csv(path)
        out[key] = df
    return out


def _day_index(frame: pd.DataFrame, start_col: str, end_col: str | None) -> pd.DatetimeIndex:
    if frame.empty:
        return pd.DatetimeIndex([])
    start = pd.to_datetime(frame[start_col])
    end = pd.to_datetime(frame[end_col]) if end_col and end_col in frame.columns else start
    days: list[pd.Timestamp] = []
    for s, e in zip(start, end):
        days.extend(pd.date_range(s.normalize(), e.normalize(), freq="D"))
    return pd.DatetimeIndex(sorted(set(days)))


def external_days(external: dict) -> dict:
    """Day-level lookup tables used by the audit."""
    holiday_days = _day_index(external["holidays"], "date", None)
    festive_days = _day_index(external["festive"], "date", None)
    event_days = _day_index(external["events"], "date_start", "date_end")
    school_days = _day_index(external["school"], "date_start", "date_end")
    return {
        "holidays": holiday_days,
        "festive": festive_days,
        "events": event_days,
        "school": school_days,
    }


def classify_event(raw_values: pd.Series, jump_threshold: float) -> str:
    """Section 7 rules, applied in the handover's order."""
    vals = raw_values.dropna().astype(float)
    if vals.empty:
        return "sensor-suspect"
    if (vals <= 0).any() or (vals >= 5300).any():
        return "sensor-suspect"
    if len(vals) >= 6 and vals.rolling(6).std().fillna(np.inf).min() < 1:
        return "sensor-suspect"
    if vals.diff().abs().max() > jump_threshold:
        return "sensor-suspect"
    return "demand anomaly"


def _match_event_name(external: dict, day: pd.Timestamp) -> tuple[str, str] | None:
    events = external["events"]
    if events.empty:
        return None
    for _, row in events.iterrows():
        start = pd.to_datetime(row["date_start"]).normalize()
        end = pd.to_datetime(row["date_end"]).normalize()
        if start <= day <= end:
            return str(row["event_name"]), str(row.get("source_url", ""))
    return None


def _rain_at(weather: pd.DataFrame, timestamp: pd.Timestamp, mm_per_hour: float = 10.0) -> bool:
    """Heavy rain within +/-2 h of the event."""
    if weather.empty:
        return False
    ts = pd.to_datetime(weather["timestamp"])
    window = ts[(ts >= timestamp - pd.Timedelta(hours=2)) & (ts <= timestamp + pd.Timedelta(hours=2))]
    if window.empty:
        return False
    values = pd.to_numeric(weather.loc[window.index, "precipitation_mm"], errors="coerce").fillna(0.0)
    return bool((values >= mm_per_hour).any())


def audit_events(
    events: pd.DataFrame,
    raw_series: pd.Series | None = None,
    jump_threshold: float | None = None,
    external: dict | None = None,
    days: dict | None = None,
) -> pd.DataFrame:
    """Section 8.2: assign exactly one category per event, by priority."""
    if events.empty:
        out = events.copy()
        out["category"] = pd.Series(dtype="object")
        out["matched_name"] = pd.Series(dtype="object")
        out["source_url"] = pd.Series(dtype="object")
        out["rule_detail"] = pd.Series(dtype="object")
        return out
    external = external if external is not None else load_external()
    days = days if days is not None else external_days(external)
    weather = external["weather"]
    interpolated = set()
    if raw_series is not None:
        interpolated = set(raw_series[raw_series.isna()].index)

    cats: list[str] = []
    names: list[str] = []
    urls: list[str] = []
    details: list[str] = []
    weekend_flags: list[bool] = []
    for _, event in events.iterrows():
        start = pd.Timestamp(event["start"])
        end = pd.Timestamp(event["end"])
        day = start.normalize()
        values = raw_series.loc[start:end] if raw_series is not None else pd.Series(dtype=float)
        name = ""
        url = ""
        detail = ""

        # rule 1: sensor-suspect, including events overlapping missing/interpolated data
        if raw_series is not None and jump_threshold is not None:
            category = classify_event(values, jump_threshold)
            detail = "Section 7 rules"
        else:
            category = "unexplained"
            detail = "no raw series"
        overlaps_gap = any(ts in interpolated for ts in pd.date_range(start, end, freq="10min"))
        if overlaps_gap:
            category = "sensor-suspect"
            detail = "overlaps interpolated or missing data"

        # rule 2: KLCC event
        if category != "sensor-suspect":
            matched = _match_event_name(external, day)
            if matched:
                category = "KLCC event"
                name, url = matched[0], matched[1]
                detail = "event day"

        # rule 3: public holiday, with adjacent long-weekend effect
        if category not in ("sensor-suspect", "KLCC event"):
            if day in days["holidays"]:
                category = "public holiday"
                name = "public holiday"
                detail = "exact day"
            elif (day - pd.Timedelta(days=1)) in days["holidays"] or (day + pd.Timedelta(days=1)) in days["holidays"]:
                category = "public holiday"
                name = "public holiday"
                detail = "adjacent to holiday"

        # rule 4: festive period or school holiday
        if category not in ("sensor-suspect", "KLCC event", "public holiday"):
            if day in days["festive"]:
                category = "festive period"
                name = "festive period"
                detail = "festive day"
            elif day in days["school"]:
                category = "school holiday"
                name = "school holiday"
                detail = "school holiday day"

        # rule 5: heavy rain
        if category not in ("sensor-suspect", "KLCC event", "public holiday", "festive period", "school holiday"):
            if _rain_at(weather, start):
                category = "heavy rain"
                name = "heavy rain"
                detail = ">=10 mm/h within 2 h"

        # rule 6 was a weekend category; the addendum (Fix 7) removes it as an
        # explanation on its own (weekends are ~29% of days) and merges it into
        # unexplained, keeping a separate on_weekend flag so the day-level
        # base-rate test can still quantify over-representation
        if category in ("demand anomaly", "unexplained"):
            category = "unexplained"
            name = ""
            if day.day_of_week >= 5:
                detail = "no external explanation (weekend)"
                on_weekend = True
            else:
                detail = "no external explanation"
                on_weekend = False
        else:
            on_weekend = day.day_of_week >= 5

        cats.append(category)
        names.append(name)
        urls.append(url)
        details.append(detail)
        weekend_flags.append(bool(on_weekend))
    out = events.copy()
    out["category"] = cats
    out["matched_name"] = names
    out["source_url"] = urls
    out["rule_detail"] = details
    out["on_weekend"] = weekend_flags
    return out


def explained_day_index(timestamps: pd.DatetimeIndex, external: dict, days: dict) -> pd.DatetimeIndex:
    """Days considered 'explained' for p0: event, holiday or festive days."""
    explained = set(days["holidays"]) | set(days["festive"]) | set(days["events"])
    if not explained:
        return pd.DatetimeIndex([])
    return pd.DatetimeIndex(sorted(day for day in explained if day in set(timestamps.normalize())))
