from __future__ import annotations

from typing import Any, Iterable

import numpy as np
import pandas as pd
from statsmodels.tsa.seasonal import STL

try:
    from statsmodels.tsa.seasonal import MSTL
except ImportError:  # pragma: no cover
    MSTL = None


def _strength_from_components(component: Iterable[float], remainder: Iterable[float]) -> float | None:
    component_values = np.asarray(component, dtype=float)
    remainder_values = np.asarray(remainder, dtype=float)
    if component_values.size == 0 or remainder_values.size == 0:
        return None
    denominator = np.var(component_values + remainder_values)
    if denominator <= 0:
        return None
    return float(max(0.0, 1.0 - np.var(remainder_values) / denominator))


def analyze_trend_and_decomposition(
    values: pd.Series,
    candidate_periods: tuple[int, ...],
    primary_period: int | None,
) -> tuple[dict[str, Any], Any]:
    numeric = pd.to_numeric(values, errors="coerce").dropna().astype(float)
    if len(numeric) < 16:
        return {
            "decomposition_method": "none",
            "trend_strength": None,
            "seasonal_strength": None,
            "seasonality_strengths": {},
        }, None

    valid_periods = [period for period in candidate_periods if period >= 2 and len(numeric) >= period * 2]
    decomposition = None
    method = "none"
    trend_strength = None
    seasonal_strength = None
    seasonal_strengths: dict[str, float] = {}

    try:
        if MSTL is not None and len(valid_periods) >= 2:
            decomposition = MSTL(numeric, periods=valid_periods[:3]).fit()
            method = "mstl"
            seasonal_component = decomposition.seasonal.sum(axis=1)
            trend_strength = _strength_from_components(decomposition.trend, decomposition.resid)
            seasonal_strength = _strength_from_components(seasonal_component, decomposition.resid)
            for index, period in enumerate(valid_periods[:3]):
                seasonal_strengths[str(period)] = _strength_from_components(decomposition.seasonal.iloc[:, index], decomposition.resid)
        elif primary_period and len(numeric) >= primary_period * 2:
            decomposition = STL(numeric, period=primary_period, robust=True).fit()
            method = "stl"
            trend_strength = _strength_from_components(decomposition.trend, decomposition.resid)
            seasonal_strength = _strength_from_components(decomposition.seasonal, decomposition.resid)
            seasonal_strengths[str(primary_period)] = seasonal_strength
    except Exception:
        decomposition = None
        method = "failed"

    return {
        "decomposition_method": method,
        "trend_strength": trend_strength,
        "seasonal_strength": seasonal_strength,
        "seasonality_strengths": seasonal_strengths,
    }, decomposition


def detect_level_shifts(values: pd.Series, timestamps: pd.Series) -> list[dict[str, Any]]:
    try:
        import ruptures as rpt
    except ImportError:
        return []

    numeric = np.asarray(pd.to_numeric(values, errors="coerce").fillna(0.0), dtype=float)
    if len(numeric) < 40:
        return []

    signal = np.log1p(np.clip(numeric, 0, None)).reshape(-1, 1)
    try:
        algo = rpt.Pelt(model="rbf").fit(signal)
        penalty = max(np.log(len(signal)) * np.var(signal), 1.0)
        breakpoints = algo.predict(pen=penalty)
    except Exception:
        return []

    return [
        {"label": "detected_level_shift", "timestamp": pd.Timestamp(timestamps.iloc[index - 1])}
        for index in breakpoints[:-1]
        if 0 < index <= len(timestamps)
    ]
