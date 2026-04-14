from __future__ import annotations

import math
from typing import Iterable

import numpy as np
import pandas as pd

try:
    import holidays
except ImportError:  # pragma: no cover - depends on runtime environment
    holidays = None

from metro_bike_share_forecasting.system_level.forecasting.config import SystemLevelConfig


def _season_tag(month: pd.Series) -> pd.Series:
    return pd.Series(
        np.select(
            [
                month.isin([12, 1, 2]),
                month.isin([3, 4, 5]),
                month.isin([6, 7, 8]),
                month.isin([9, 10, 11]),
            ],
            ["winter", "spring", "summer", "fall"],
            default="unknown",
        ),
        index=month.index,
    )


def _fourier_terms(index: np.ndarray, period: float, order: int, prefix: str) -> dict[str, np.ndarray]:
    values: dict[str, np.ndarray] = {}
    for harmonic in range(1, order + 1):
        angle = 2 * math.pi * harmonic * index / period
        values[f"{prefix}_sin_{harmonic}"] = np.sin(angle)
        values[f"{prefix}_cos_{harmonic}"] = np.cos(angle)
    return values


def _calendar_position(date_series: pd.Series) -> np.ndarray:
    origin = pd.Timestamp("2000-01-01")
    return (pd.to_datetime(date_series) - origin).dt.days.to_numpy(dtype=float)


def build_known_future_features(date_frame: pd.DataFrame, config: SystemLevelConfig, external_features: pd.DataFrame) -> pd.DataFrame:
    known = date_frame[["date"]].copy()
    known["day_of_week"] = known["date"].dt.dayofweek
    known["is_weekend"] = known["day_of_week"].isin([5, 6]).astype(int)
    known["week_of_year"] = known["date"].dt.isocalendar().week.astype(int)
    known["month"] = known["date"].dt.month
    known["quarter"] = known["date"].dt.quarter
    known["year"] = known["date"].dt.year
    known["day_of_month"] = known["date"].dt.day
    known["day_of_year"] = known["date"].dt.dayofyear
    known["season_tag"] = _season_tag(known["month"])
    season_map = {"winter": 0, "spring": 1, "summer": 2, "fall": 3, "unknown": -1}
    known["season_code"] = known["season_tag"].map(season_map).astype(int)

    if holidays is not None:
        holiday_calendar = holidays.country_holidays(config.holiday_country)
        known["is_holiday"] = known["date"].dt.date.map(lambda value: int(value in holiday_calendar))
    else:
        known["is_holiday"] = 0

    day_index = _calendar_position(known["date"])
    if config.include_weekly_fourier:
        for name, values in _fourier_terms(day_index, 7.0, config.weekly_fourier_order, "weekly").items():
            known[name] = values
    if config.include_yearly_fourier:
        for name, values in _fourier_terms(day_index, 365.25, config.yearly_fourier_order, "yearly").items():
            known[name] = values

    if not external_features.empty:
        known = known.merge(external_features, on="date", how="left")
    return known


def build_system_level_features(
    target_frame: pd.DataFrame,
    config: SystemLevelConfig,
    external_features: pd.DataFrame,
) -> pd.DataFrame:
    known = build_known_future_features(target_frame[["date"]], config, external_features)
    frame = target_frame.merge(known, on="date", how="left")

    for lag in config.lags:
        frame[f"lag_{lag}"] = frame["target"].shift(lag)

    for window in config.rolling_windows:
        shifted = frame["target"].shift(1)
        frame[f"rolling_mean_{window}"] = shifted.rolling(window, min_periods=max(2, min(window, 7))).mean()
        frame[f"rolling_std_{window}"] = shifted.rolling(window, min_periods=max(2, min(window, 7))).std()
        frame[f"rolling_min_{window}"] = shifted.rolling(window, min_periods=max(2, min(window, 7))).min()
        frame[f"rolling_max_{window}"] = shifted.rolling(window, min_periods=max(2, min(window, 7))).max()

    return frame


def known_future_feature_columns(frame: pd.DataFrame) -> list[str]:
    excluded = {"date", "target", "missing_period_flag", "series_scope", "season_tag"}
    lag_like = {column for column in frame.columns if column.startswith("lag_") or column.startswith("rolling_")}
    return [column for column in frame.columns if column not in excluded and column not in lag_like]


def ml_feature_columns(frame: pd.DataFrame) -> list[str]:
    excluded = {"date", "target", "missing_period_flag", "series_scope", "season_tag"}
    columns = [column for column in frame.columns if column not in excluded]
    numeric = frame[columns].select_dtypes(include=["number", "bool"]).columns.tolist()
    return numeric


def build_future_dates(last_date: pd.Timestamp, horizon: int) -> pd.DataFrame:
    return pd.DataFrame({"date": pd.date_range(last_date + pd.Timedelta(days=1), periods=horizon, freq="D")})


def rebuild_feature_frame_with_predictions(
    history_target: pd.DataFrame,
    config: SystemLevelConfig,
    external_features: pd.DataFrame,
) -> pd.DataFrame:
    return build_system_level_features(history_target.copy(), config, external_features)
