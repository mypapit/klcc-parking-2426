"""Tests for the Section 8 audit and the base-rate helpers."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from audit import audit_events, classify_event, load_external, _day_index  # noqa: E402


def _external(overrides: dict) -> dict:
    base = load_external()
    base.update(overrides)
    return base


def test_classify_event_zero_is_sensor():
    series = pd.Series([100, 0, 120])
    assert classify_event(series, jump_threshold=5000) == "sensor-suspect"


def test_classify_event_full_is_sensor():
    series = pd.Series([100, 5300, 120])
    assert classify_event(series, jump_threshold=5000) == "sensor-suspect"


def test_classify_event_flat_is_sensor():
    series = pd.Series([120.0] * 8)
    assert classify_event(series, jump_threshold=5000) == "sensor-suspect"


def test_classify_event_jump_is_sensor():
    series = pd.Series([100.0, 100.0, 900.0, 100.0])
    assert classify_event(series, jump_threshold=100) == "sensor-suspect"


def test_classify_event_normal_is_demand():
    series = pd.Series([100.0, 120.0, 110.0, 130.0])
    assert classify_event(series, jump_threshold=1000) == "demand anomaly"


def test_audit_priority_event_over_weekend():
    events = pd.DataFrame([{"start": "2023-06-29 10:00", "end": "2023-06-29 10:30", "score": 1.0}])
    # 2023-06-29 is a weekday; give it an event
    external = _external({
        "events": pd.DataFrame([{
            "date_start": "2023-06-29", "date_end": "2023-06-30",
            "event_name": "Test Expo", "venue": "KLCC", "source_url": "http://example.com",
        }])
    })
    raw = pd.Series(100.0, index=pd.date_range("2023-06-29 10:00", periods=4, freq="10min"))
    audited = audit_events(events, raw, jump_threshold=1000, external=external)
    assert audited.iloc[0]["category"] == "KLCC event"
    assert audited.iloc[0]["source_url"] == "http://example.com"


def test_audit_weekend_when_no_other_cause():
    # Fix 7: a weekend is no longer a category of its own; it is merged into
    # unexplained and flagged in the on_weekend column
    events = pd.DataFrame([{"start": "2023-07-01 10:00", "end": "2023-07-01 10:30", "score": 1.0}])
    external = _external({
        "events": pd.DataFrame(columns=["date_start", "date_end", "event_name", "source_url"]),
        "weather": pd.DataFrame(columns=["timestamp", "precipitation_mm"]),
    })
    raw = pd.Series(100.0, index=pd.date_range("2023-07-01 10:00", periods=4, freq="10min"))
    audited = audit_events(events, raw, jump_threshold=1000, external=external)
    assert audited.iloc[0]["category"] == "unexplained"
    assert bool(audited.iloc[0]["on_weekend"]) is True


def test_audit_unexplained_when_no_cause():
    events = pd.DataFrame([{"start": "2023-07-03 10:00", "end": "2023-07-03 10:30", "score": 1.0}])
    external = _external({
        "events": pd.DataFrame(columns=["date_start", "date_end", "event_name", "source_url"]),
        "weather": pd.DataFrame(columns=["timestamp", "precipitation_mm"]),
    })
    raw = pd.Series(100.0, index=pd.date_range("2023-07-03 10:00", periods=4, freq="10min"))
    audited = audit_events(events, raw, jump_threshold=1000, external=external)
    assert audited.iloc[0]["category"] == "unexplained"


def test_day_index_expands_ranges():
    frame = pd.DataFrame([{"date_start": "2023-01-01", "date_end": "2023-01-03"}])
    days = _day_index(frame, "date_start", "date_end")
    assert len(days) == 3
    assert pd.Timestamp("2023-01-02") in days


def test_day_index_empty():
    days = _day_index(pd.DataFrame(columns=["date_start", "date_end"]), "date_start", "date_end")
    assert len(days) == 0
