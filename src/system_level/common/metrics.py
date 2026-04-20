from __future__ import annotations

import numpy as np
import pandas as pd


FREQUENCY_TO_SEASON_LENGTH = {
    "h": 24,
    "hour": 24,
    "hourly": 24,
    "d": 7,
    "day": 7,
    "daily": 7,
    "w": 52,
    "week": 52,
    "weekly": 52,
    "m": 12,
    "month": 12,
    "monthly": 12,
    "q": 4,
    "quarter": 4,
    "quarterly": 4,
    "y": 1,
    "year": 1,
    "yearly": 1,
}


def default_mase_season_length(frequency: str | None, fallback: int = 1) -> int:
    if frequency is None:
        return int(fallback)
    normalized = str(frequency).strip().lower()
    return int(FREQUENCY_TO_SEASON_LENGTH.get(normalized, fallback))


def mae(actual: pd.Series, predicted: pd.Series) -> float:
    return float(np.mean(np.abs(actual - predicted)))


def rmse(actual: pd.Series, predicted: pd.Series) -> float:
    return float(np.sqrt(np.mean(np.square(actual - predicted))))


def bias(actual: pd.Series, predicted: pd.Series) -> float:
    return float(np.mean(predicted - actual))


def seasonal_naive_scale(training_series: pd.Series, season_length: int) -> float:
    training_array = pd.Series(training_series).dropna().to_numpy(dtype=float)
    effective_season_length = max(int(season_length), 1)
    if len(training_array) <= effective_season_length:
        naive_errors = np.abs(np.diff(training_array)) if len(training_array) > 1 else np.array([1.0])
    else:
        naive_errors = np.abs(training_array[effective_season_length:] - training_array[:-effective_season_length])
    scale = float(np.mean(naive_errors)) if len(naive_errors) else 1.0
    return scale if np.isfinite(scale) and scale > 0 else 1.0


def mase(actual: pd.Series, predicted: pd.Series, training_series: pd.Series, season_length: int) -> float:
    scale = seasonal_naive_scale(training_series, season_length)
    return float(np.mean(np.abs(actual - predicted)) / scale)
