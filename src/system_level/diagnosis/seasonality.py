from __future__ import annotations

from typing import Any, Iterable

import numpy as np
import pandas as pd


def choose_primary_period(candidate_periods: Iterable[int], series_length: int, fallback: int | None) -> int | None:
    for period in candidate_periods:
        if period >= 2 and series_length >= period * 2:
            return int(period)
    if fallback and fallback >= 2 and series_length >= fallback * 2:
        return int(fallback)
    return None


def detect_multiple_seasonality(frequency_summary: dict[str, Any], candidate_periods: tuple[int, ...]) -> bool:
    dominant_periods = list(frequency_summary.get("dominant_periods", []))
    matched_candidate_periods = list(frequency_summary.get("matched_candidate_periods", []))
    strong_primary = bool(frequency_summary.get("strong_primary_spectral_peak", False))
    if len(dominant_periods) < 2 or not strong_primary:
        return False
    if len(set(matched_candidate_periods)) >= 2:
        return True
    return len(dominant_periods) >= 3 and len(candidate_periods) >= 2


def build_profile_tables(prepared: pd.DataFrame, frequency: str | None) -> dict[str, pd.DataFrame]:
    timestamp = prepared["timestamp"]
    profiles: dict[str, pd.DataFrame] = {}

    weekday_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    weekday_profile = (
        prepared.assign(weekday=timestamp.dt.day_name())
        .groupby("weekday", as_index=False)
        .agg(
            average_observed_value=("observed_value", "mean"),
            average_filled_value=("value_filled", "mean"),
            observed_points=("observed_value", lambda series: int(series.notna().sum())),
            imputed_points=("imputed_flag", "sum"),
        )
    )
    weekday_profile["weekday"] = pd.Categorical(weekday_profile["weekday"], categories=weekday_order, ordered=True)
    profiles["weekday_profile"] = weekday_profile.sort_values("weekday").reset_index(drop=True)

    month_order = [
        "January","February","March","April","May","June",
        "July","August","September","October","November","December",
    ]
    monthly_profile = (
        prepared.assign(month=timestamp.dt.month_name())
        .groupby("month", as_index=False)
        .agg(
            average_observed_value=("observed_value", "mean"),
            average_filled_value=("value_filled", "mean"),
            observed_points=("observed_value", lambda series: int(series.notna().sum())),
            imputed_points=("imputed_flag", "sum"),
        )
    )
    monthly_profile["month"] = pd.Categorical(monthly_profile["month"], categories=month_order, ordered=True)
    profiles["monthly_profile"] = monthly_profile.sort_values("month").reset_index(drop=True)

    if (frequency or "").lower() == "hourly":
        intraday_profile = (
            prepared.assign(
                hour=timestamp.dt.hour,
                weekend=np.where(timestamp.dt.dayofweek >= 5, "weekend", "weekday"),
            )
            .groupby(["weekend", "hour"], as_index=False)
            .agg(
                average_observed_value=("observed_value", "mean"),
                average_filled_value=("value_filled", "mean"),
                observed_points=("observed_value", lambda series: int(series.notna().sum())),
                imputed_points=("imputed_flag", "sum"),
            )
        )
        profiles["intraday_profile"] = intraday_profile

    return profiles
