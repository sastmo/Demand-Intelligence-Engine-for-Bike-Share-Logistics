from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from metro_bike_share_forecasting.station_level.forecasting.config import StationLevelForecastConfig


@dataclass(frozen=True)
class StationBacktestWindow:
    fold_id: int
    train_end: pd.Timestamp
    test_end: pd.Timestamp
    forecast_dates: pd.DatetimeIndex


def build_station_backtest_windows(panel: pd.DataFrame, config: StationLevelForecastConfig) -> list[StationBacktestWindow]:
    observed_dates = sorted(pd.to_datetime(panel.loc[panel["in_service"], "date"]).dropna().unique().tolist())
    if len(observed_dates) <= config.initial_train_size + config.max_backtest_horizon:
        return []

    windows: list[StationBacktestWindow] = []
    train_end_index = config.initial_train_size - 1
    max_horizon = config.max_backtest_horizon
    while train_end_index + max_horizon < len(observed_dates):
        train_end = pd.Timestamp(observed_dates[train_end_index])
        forecast_dates = pd.date_range(train_end + pd.Timedelta(days=1), periods=max_horizon, freq="D")
        windows.append(
            StationBacktestWindow(
                fold_id=len(windows) + 1,
                train_end=train_end,
                test_end=pd.Timestamp(forecast_dates.max()),
                forecast_dates=forecast_dates,
            )
        )
        train_end_index += config.step_size

    if len(windows) > config.max_folds:
        windows = windows[-config.max_folds :]
        windows = [StationBacktestWindow(index + 1, window.train_end, window.test_end, window.forecast_dates) for index, window in enumerate(windows)]
    return windows


def station_mase_scales(train_panel: pd.DataFrame) -> dict[str, float]:
    scales: dict[str, float] = {}
    observed = train_panel.loc[train_panel["in_service"]].sort_values(["station_id", "date"])
    for station_id, station_frame in observed.groupby("station_id", sort=True):
        values = station_frame["target"].astype(float).dropna().to_numpy()
        if len(values) <= 7:
            diffs = np.abs(np.diff(values)) if len(values) > 1 else np.array([1.0])
        else:
            diffs = np.abs(values[7:] - values[:-7])
        scale = float(np.mean(diffs)) if len(diffs) else 1.0
        scales[str(station_id)] = scale if np.isfinite(scale) and scale > 0 else 1.0
    return scales


def expand_horizon_rows(
    scored_forecasts: pd.DataFrame,
    train_panel: pd.DataFrame,
    config: StationLevelForecastConfig,
) -> pd.DataFrame:
    if scored_forecasts.empty:
        return scored_forecasts.copy()

    scales = station_mase_scales(train_panel)
    rows: list[pd.DataFrame] = []
    for horizon in config.forecast_horizons:
        subset = scored_forecasts.loc[scored_forecasts["horizon_step"] <= horizon].copy()
        subset["horizon"] = horizon
        rows.append(subset)

    expanded = pd.concat(rows, ignore_index=True)
    expanded["abs_error"] = (expanded["prediction"] - expanded["actual"]).abs()
    expanded["squared_error"] = (expanded["prediction"] - expanded["actual"]) ** 2
    expanded["bias_error"] = expanded["prediction"] - expanded["actual"]
    expanded["mase_scale"] = expanded["station_id"].astype(str).map(scales).fillna(1.0)
    expanded["scaled_abs_error"] = expanded["abs_error"] / expanded["mase_scale"].replace(0.0, 1.0)
    return expanded


def summarize_fold_metrics(scored_forecasts: pd.DataFrame) -> pd.DataFrame:
    if scored_forecasts.empty:
        return pd.DataFrame()
    metrics = (
        scored_forecasts.groupby(["model_name", "fold_id", "horizon"], as_index=False)
        .agg(
            mae=("abs_error", "mean"),
            rmse=("squared_error", lambda values: float(np.sqrt(np.mean(values)))),
            mase=("scaled_abs_error", "mean"),
            bias=("bias_error", "mean"),
            rows=("actual", "size"),
            stations=("station_id", "nunique"),
        )
        .sort_values(["model_name", "fold_id", "horizon"])
        .reset_index(drop=True)
    )
    return metrics
