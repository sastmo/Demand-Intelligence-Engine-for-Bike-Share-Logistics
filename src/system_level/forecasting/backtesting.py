from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd

from system_level.common.cli_utils import noop_progress
from system_level.common.metrics import bias, mae, mase, rmse
from system_level.forecasting.config import SystemLevelConfig
from system_level.forecasting.models import MODEL_DIAGNOSTIC_COLUMNS, MODEL_REGISTRY


@dataclass(frozen=True)
class RollingWindow:
    fold_id: int
    horizon: int
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp
    train_frame: pd.DataFrame
    test_frame: pd.DataFrame


def build_rolling_windows(frame: pd.DataFrame, config: SystemLevelConfig, horizon: int) -> list[RollingWindow]:
    ordered = frame.sort_values("date").reset_index(drop=True)
    windows: list[RollingWindow] = []
    train_end_index = config.initial_train_size
    while train_end_index + horizon <= len(ordered):
        train = ordered.iloc[:train_end_index].copy()
        test = ordered.iloc[train_end_index : train_end_index + horizon].copy()
        windows.append(
            RollingWindow(
                fold_id=len(windows) + 1,
                horizon=horizon,
                train_start=train["date"].min(),
                train_end=train["date"].max(),
                test_start=test["date"].min(),
                test_end=test["date"].max(),
                train_frame=train,
                test_frame=test,
            )
        )
        train_end_index += config.step_size
    if len(windows) > config.max_folds:
        windows = windows[-config.max_folds :]
        windows = [
            RollingWindow(
                fold_id=index + 1,
                horizon=window.horizon,
                train_start=window.train_start,
                train_end=window.train_end,
                test_start=window.test_start,
                test_end=window.test_end,
                train_frame=window.train_frame,
                test_frame=window.test_frame,
            )
            for index, window in enumerate(windows)
        ]
    return windows


def evaluate_prediction_window(
    window: RollingWindow,
    forecast_frame: pd.DataFrame,
    config: SystemLevelConfig,
) -> dict[str, object]:
    merged = window.test_frame[["date", "target"]].merge(forecast_frame, on="date", how="left")
    return {
        "fold_id": window.fold_id,
        "horizon": window.horizon,
        "train_start": window.train_start,
        "train_end": window.train_end,
        "test_start": window.test_start,
        "test_end": window.test_end,
        "mae": mae(merged["target"], merged["prediction"]),
        "rmse": rmse(merged["target"], merged["prediction"]),
        "mase": mase(
            merged["target"],
            merged["prediction"],
            window.train_frame["target"],
            season_length=config.mase_season_length,
        ),
        "bias": bias(merged["target"], merged["prediction"]),
    }


def run_backtests(
    target_frame: pd.DataFrame,
    external_features: pd.DataFrame,
    config: SystemLevelConfig,
    model_keys: list[str],
    progress: Callable[[str], None] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    progress = progress or noop_progress
    metric_rows: list[dict[str, object]] = []
    forecast_rows: list[pd.DataFrame] = []
    window_rows: list[dict[str, object]] = []

    for horizon in config.forecast_horizons:
        windows = build_rolling_windows(target_frame, config, horizon)
        progress(f"Backtest horizon {horizon}: {len(windows)} fold(s), {len(model_keys)} model(s).")
        for window_index, window in enumerate(windows, start=1):
            progress(
                f"Horizon {horizon} fold {window_index}/{len(windows)}: "
                f"train_end={window.train_end.date()} test_end={window.test_end.date()}."
            )
            window_rows.append(
                {
                    "fold_id": window.fold_id,
                    "horizon": window.horizon,
                    "train_start": window.train_start,
                    "train_end": window.train_end,
                    "test_start": window.test_start,
                    "test_end": window.test_end,
                }
            )
            for model_index, model_key in enumerate(model_keys, start=1):
                progress(f"Horizon {horizon} fold {window_index}/{len(windows)} model {model_index}/{len(model_keys)}: {model_key}.")
                forecast_frame = MODEL_REGISTRY[model_key](window.train_frame, horizon, config, external_features).copy()
                actual_model_name = (
                    str(forecast_frame["model_name"].iloc[0]) if "model_name" in forecast_frame.columns and not forecast_frame.empty else model_key
                )
                forecast_frame = forecast_frame.sort_values("date").reset_index(drop=True)
                forecast_frame["fold_id"] = window.fold_id
                forecast_frame["horizon"] = horizon
                forecast_frame["horizon_step"] = np.arange(1, len(forecast_frame) + 1, dtype=int)
                forecast_frame["actual"] = window.test_frame["target"].to_numpy(dtype=float)
                forecast_rows.append(forecast_frame)

                metric_row = evaluate_prediction_window(window, forecast_frame, config)
                metric_row["model_name"] = actual_model_name
                for column in MODEL_DIAGNOSTIC_COLUMNS:
                    metric_row[column] = forecast_frame[column].iloc[0] if column in forecast_frame.columns and not forecast_frame.empty else None
                metric_rows.append(metric_row)

    metrics = pd.DataFrame(metric_rows)
    forecasts = pd.concat(forecast_rows, ignore_index=True) if forecast_rows else pd.DataFrame()
    windows_frame = pd.DataFrame(window_rows)
    progress(f"Backtests complete: metric_rows={len(metrics)} forecast_rows={len(forecasts)}.")
    return metrics, forecasts, windows_frame
