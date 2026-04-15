from __future__ import annotations

import math

import numpy as np
import pandas as pd

try:
    import holidays
except ImportError:  # pragma: no cover
    holidays = None

from metro_bike_share_forecasting.station_level.forecasting.config import StationLevelForecastConfig


STATION_FEATURE_COLUMNS = [
    "station_id",
    "station_age_days",
    "day_of_week",
    "month",
    "is_weekend",
    "is_holiday",
    "lag_1",
    "lag_7",
    "lag_14",
    "rolling_mean_7",
    "rolling_mean_28",
    "rolling_std_28",
]


def _holiday_flag(date_values: pd.Series, config: StationLevelForecastConfig) -> pd.Series:
    if holidays is None:
        return pd.Series(0, index=date_values.index, dtype=int)
    calendar = holidays.country_holidays(config.holiday_country)
    return date_values.dt.date.map(lambda value: int(value in calendar)).astype(int)


def build_station_feature_frame(
    panel: pd.DataFrame,
    config: StationLevelForecastConfig,
    slice_lookup: pd.DataFrame | None = None,
) -> pd.DataFrame:
    frame = panel.copy().sort_values(["station_id", "date"]).reset_index(drop=True)
    if frame.empty:
        return frame

    station_start = frame.loc[frame["in_service"]].groupby("station_id")["date"].min()
    frame["station_start_date"] = frame["station_id"].map(station_start)
    frame["station_age_days"] = (frame["date"] - frame["station_start_date"]).dt.days.clip(lower=0)
    frame["day_of_week"] = frame["date"].dt.dayofweek
    frame["month"] = frame["date"].dt.month
    frame["is_weekend"] = frame["day_of_week"].isin([5, 6]).astype(int)
    frame["is_holiday"] = _holiday_flag(frame["date"], config)

    grouped = frame.groupby("station_id", group_keys=False)
    frame["lag_1"] = grouped["target"].shift(1)
    frame["lag_7"] = grouped["target"].shift(7)
    frame["lag_14"] = grouped["target"].shift(14)
    frame["rolling_mean_7"] = grouped["target"].transform(lambda series: series.shift(1).rolling(7, min_periods=3).mean())
    frame["rolling_mean_28"] = grouped["target"].transform(lambda series: series.shift(1).rolling(28, min_periods=7).mean())
    frame["rolling_std_28"] = grouped["target"].transform(lambda series: series.shift(1).rolling(28, min_periods=7).std())

    if slice_lookup is not None and not slice_lookup.empty:
        frame = frame.merge(slice_lookup, on="station_id", how="left")
        if config.include_category_feature and "station_category" in frame.columns:
            STATION_FEATURE_COLUMNS.append("station_category")
        if config.include_cluster_feature and "cluster_label" in frame.columns:
            STATION_FEATURE_COLUMNS.append("cluster_label")

    return frame


def training_rows(feature_frame: pd.DataFrame) -> pd.DataFrame:
    return feature_frame.loc[feature_frame["in_service"] & feature_frame["target"].notna()].copy()


def station_start_dates(panel: pd.DataFrame) -> dict[str, pd.Timestamp]:
    starts = panel.loc[panel["in_service"]].groupby("station_id")["date"].min()
    return {str(key): pd.Timestamp(value) for key, value in starts.items()}


def history_lookup(panel: pd.DataFrame) -> dict[str, pd.Series]:
    lookup: dict[str, pd.Series] = {}
    ordered = panel.sort_values(["station_id", "date"])
    for station_id, station_frame in ordered.groupby("station_id", sort=True):
        lookup[str(station_id)] = station_frame.set_index("date")["target"].astype(float).copy()
    return lookup


def build_future_station_rows(
    forecast_date: pd.Timestamp,
    station_ids: list[str],
    history_by_station: dict[str, pd.Series],
    station_start_by_station: dict[str, pd.Timestamp],
    config: StationLevelForecastConfig,
    slice_lookup: pd.DataFrame | None = None,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    if slice_lookup is not None and not slice_lookup.empty:
        slice_map = slice_lookup.set_index("station_id").to_dict(orient="index")
    else:
        slice_map = {}

    for station_id in station_ids:
        history = history_by_station.get(station_id, pd.Series(dtype=float))
        station_start = station_start_by_station.get(station_id, forecast_date)
        lag_1 = history.get(forecast_date - pd.Timedelta(days=1), np.nan)
        lag_7 = history.get(forecast_date - pd.Timedelta(days=7), np.nan)
        lag_14 = history.get(forecast_date - pd.Timedelta(days=14), np.nan)

        previous_7 = history.reindex(pd.date_range(forecast_date - pd.Timedelta(days=7), periods=7, freq="D"))
        previous_28 = history.reindex(pd.date_range(forecast_date - pd.Timedelta(days=28), periods=28, freq="D"))

        row = {
            "station_id": station_id,
            "date": forecast_date,
            "in_service": True,
            "target": np.nan,
            "missing_period_flag": 0,
            "series_scope": "station_level",
            "station_age_days": max(int((forecast_date - station_start).days), 0),
            "day_of_week": int(forecast_date.dayofweek),
            "month": int(forecast_date.month),
            "is_weekend": int(forecast_date.dayofweek in {5, 6}),
            "lag_1": float(lag_1) if pd.notna(lag_1) else np.nan,
            "lag_7": float(lag_7) if pd.notna(lag_7) else np.nan,
            "lag_14": float(lag_14) if pd.notna(lag_14) else np.nan,
            "rolling_mean_7": float(previous_7.mean()) if previous_7.notna().sum() >= 3 else np.nan,
            "rolling_mean_28": float(previous_28.mean()) if previous_28.notna().sum() >= 7 else np.nan,
            "rolling_std_28": float(previous_28.std(ddof=0)) if previous_28.notna().sum() >= 7 else np.nan,
            "is_holiday": int(_holiday_flag(pd.Series([forecast_date]), config).iloc[0]),
        }
        row.update(slice_map.get(station_id, {}))
        rows.append(row)

    return pd.DataFrame(rows)


def build_future_dates(last_date: pd.Timestamp, horizon: int) -> pd.DatetimeIndex:
    return pd.date_range(last_date + pd.Timedelta(days=1), periods=horizon, freq="D")
