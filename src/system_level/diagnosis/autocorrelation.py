from __future__ import annotations

from typing import Any

import pandas as pd


def summarize_autocorrelation(values: pd.Series, primary_period: int | None) -> dict[str, Any]:
    numeric = pd.to_numeric(values, errors="coerce").dropna().astype(float)
    if len(numeric) < 3:
        return {
            "autocorrelation_status": "insufficient_history",
            "lag1_autocorrelation": None,
            "lag2_autocorrelation": None,
            "seasonal_lag_autocorrelation": None,
        }
    lag1 = float(numeric.autocorr(lag=1)) if len(numeric) > 2 else None
    lag2 = float(numeric.autocorr(lag=2)) if len(numeric) > 3 else None
    seasonal_lag = None
    if primary_period and len(numeric) > primary_period + 2:
        seasonal_lag = float(numeric.autocorr(lag=primary_period))
    return {
        "autocorrelation_status": "ok",
        "lag1_autocorrelation": lag1,
        "lag2_autocorrelation": lag2,
        "seasonal_lag_autocorrelation": seasonal_lag,
    }
