from __future__ import annotations

from typing import Any, Iterable

import numpy as np
import pandas as pd
from statsmodels.tsa.seasonal import STL

try:
    from statsmodels.tsa.seasonal import MSTL
except ImportError:  # pragma: no cover
    MSTL = None


def _load_ruptures():
    try:
        import ruptures as rpt
    except ImportError:
        return None
    return rpt


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
    mstl_available = MSTL is not None
    if len(numeric) < 16:
        return {
            "decomposition_method": "none",
            "decomposition_status": "insufficient_history",
            "decomposition_reason": "At least 16 observations are required before decomposition is attempted.",
            "decomposition_periods_used": [],
            "mstl_available": mstl_available,
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
    decomposition_status = "not_run"
    decomposition_reason = "No valid seasonal periods were available for decomposition."
    periods_used: list[int] = []

    try:
        if MSTL is not None and len(valid_periods) >= 2:
            periods_used = valid_periods[:3]
            decomposition = MSTL(numeric, periods=periods_used).fit()
            method = "mstl"
            decomposition_status = "ok"
            decomposition_reason = "MSTL was selected because multiple viable seasonal periods were available."
            seasonal_component = decomposition.seasonal.sum(axis=1)
            trend_strength = _strength_from_components(decomposition.trend, decomposition.resid)
            seasonal_strength = _strength_from_components(seasonal_component, decomposition.resid)
            for index, period in enumerate(periods_used):
                seasonal_strengths[str(period)] = _strength_from_components(decomposition.seasonal.iloc[:, index], decomposition.resid)
        elif primary_period and len(numeric) >= primary_period * 2:
            periods_used = [primary_period]
            decomposition = STL(numeric, period=primary_period, robust=True).fit()
            method = "stl"
            decomposition_status = "ok"
            decomposition_reason = "STL was selected because one viable primary seasonal period was available."
            trend_strength = _strength_from_components(decomposition.trend, decomposition.resid)
            seasonal_strength = _strength_from_components(decomposition.seasonal, decomposition.resid)
            seasonal_strengths[str(primary_period)] = seasonal_strength
        else:
            decomposition_status = "not_run"
            decomposition_reason = "No candidate seasonal period had enough history for decomposition."
    except Exception:
        decomposition = None
        method = "failed"
        decomposition_status = "failed"
        decomposition_reason = "Seasonal decomposition failed unexpectedly."

    return {
        "decomposition_method": method,
        "decomposition_status": decomposition_status,
        "decomposition_reason": decomposition_reason,
        "decomposition_periods_used": periods_used,
        "mstl_available": mstl_available,
        "trend_strength": trend_strength,
        "seasonal_strength": seasonal_strength,
        "seasonality_strengths": seasonal_strengths,
    }, decomposition


def detect_level_shifts(values: pd.Series, timestamps: pd.Series) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rpt = _load_ruptures()
    if rpt is None:
        return [], {
            "level_shift_detection_available": False,
            "level_shift_detection_status": "unavailable_optional_dependency",
            "level_shift_detection_method": None,
            "level_shift_detection_reason": "Optional dependency `ruptures` is not installed.",
        }

    numeric = np.asarray(pd.to_numeric(values, errors="coerce").fillna(0.0), dtype=float)
    if len(numeric) < 40:
        return [], {
            "level_shift_detection_available": True,
            "level_shift_detection_status": "insufficient_history",
            "level_shift_detection_method": "ruptures_pelt_rbf",
            "level_shift_detection_reason": "At least 40 observations are required before level-shift screening is attempted.",
        }

    signal = np.log1p(np.clip(numeric, 0, None)).reshape(-1, 1)
    try:
        algo = rpt.Pelt(model="rbf").fit(signal)
        penalty = max(np.log(len(signal)) * np.var(signal), 1.0)
        breakpoints = algo.predict(pen=penalty)
    except Exception as exc:
        return [], {
            "level_shift_detection_available": True,
            "level_shift_detection_status": "failed",
            "level_shift_detection_method": "ruptures_pelt_rbf",
            "level_shift_detection_reason": f"Level-shift detection failed with {type(exc).__name__}.",
        }

    level_shifts = [
        {"label": "detected_level_shift", "timestamp": pd.Timestamp(timestamps.iloc[index - 1])}
        for index in breakpoints[:-1]
        if 0 < index <= len(timestamps)
    ]
    return level_shifts, {
        "level_shift_detection_available": True,
        "level_shift_detection_status": "ok",
        "level_shift_detection_method": "ruptures_pelt_rbf",
        "level_shift_detection_reason": "Level-shift screening completed.",
    }
