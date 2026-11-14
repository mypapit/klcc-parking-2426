"""External-event audit scaffolding."""
from __future__ import annotations

import pandas as pd


def load_public_holidays(path: str) -> pd.DataFrame:
    return pd.read_csv(path, parse_dates=["date"])


def write_public_holidays(path, years=(2022, 2023, 2024)):
    import holidays

    cal = holidays.country_holidays("MY", subdiv="14", years=list(years))
    df = pd.DataFrame({"date": list(cal.keys()), "name": list(cal.values())}).sort_values("date")
    df.to_csv(path, index=False)
    return df
